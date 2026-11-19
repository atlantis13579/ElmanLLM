#include "Backends/Metal.h"
#include "Backends/HostWords.h"
#include "Core/AllocationPool.h"
#include "Backends/MPSGraph.h"
#include "Operators/attributes.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstring>
#include <deque>
#include <list>
#include <mutex>
#include <stdexcept>
#include <unordered_map>

namespace elman {
namespace {
class MetalBuffer final : public Buffer {
public:
    id<MTLBuffer> storage;
    MetalBuffer(id<MTLDevice> device, size_t count, DType dtype) : Buffer(Backend::Metal, dtype, count) {
        storage = [device newBufferWithLength:std::max(bytes, size_t(1)) options:MTLResourceStorageModeShared];
        if (!storage) throw AllocationFailure("Cannot allocate Metal tensor storage");
    }
    int device_ordinal() const override { return 0; }
    void* raw_pointer() override { return storage.contents; }
    const void* raw_pointer() const override { return storage.contents; }
};

const MetalBuffer& storage_for(const BufferPtr& value) {
    auto* storage = value ? dynamic_cast<const MetalBuffer*>(&value->allocation()) : nullptr;
    if (!storage) throw std::invalid_argument("Expected Metal tensor storage");
    return *storage;
}
size_t offset_for(const BufferPtr& value, const MetalBuffer& owner) {
    return static_cast<const uint8_t*>(value->raw_pointer()) - static_cast<const uint8_t*>(owner.raw_pointer());
}

bool random_operation(const std::string& op) {
    return op == "uniform" || op == "normal" || op == "dropout";
}

MPSMatrixRandomDistributionDescriptor* distribution_for(const std::string& op, const KernelParams& p) {
    const auto dtype = DType(p.output_dtype());
    if (!is_compute_float(dtype) && !is_low_precision_float(dtype)) return nil;
    if (op == "dropout") return metal::dropout_distribution(p);
    if (op == "normal") return metal::normal_distribution(p);
    return metal::uniform_distribution(p);
}

struct Key {
    std::string op;
    KernelParams params;
    bool operator==(const Key& other) const {
        return op == other.op && std::memcmp(params.v, other.params.v, sizeof(params.v)) == 0;
    }
};
struct KeyHash {
    size_t operator()(const Key& key) const {
        size_t hash = std::hash<std::string>{}(key.op);
        for (int value : key.params.v) hash = (hash ^ static_cast<uint32_t>(value)) * 1099511628211ull;
        return hash;
    }
};

struct Plan {
    MPSGraphExecutable* executable = nil;
    std::array<int, 4> slots{};
    size_t count = 0;

    Plan(MPSGraphDevice* device, const Key& key, bool has_samples) {
        const auto& p = key.params;
        MPSGraph* graph = [MPSGraph new];
        metal::Operands inputs{};
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds = [NSMutableDictionary dictionary];
        for (int slot = 0; slot < 4; ++slot) {
            const auto input = static_cast<KernelInput>(slot);
            if (p.input_rank(input) < 0) continue;
            auto dims = metal::storage_shape(p, slot);
            if ([dims containsObject:@0]) continue;
            const auto type = metal::storage_type(DType(p.input_dtype(input)));
            if (type == MPSDataTypeInvalid) throw std::invalid_argument("Metal dtype unavailable in " + key.op);
            inputs[slot] = [graph placeholderWithShape:dims dataType:type name:nil];
            feeds[inputs[slot]] = [[MPSGraphShapedType alloc] initWithShape:dims dataType:type];
        }
        MPSGraphTensor* samples = nil;
        if (has_samples) {
            auto dims = metal::shape(p);
            samples = [graph placeholderWithShape:dims dataType:MPSDataTypeFloat32 name:nil];
            feeds[samples] = [[MPSGraphShapedType alloc] initWithShape:dims dataType:MPSDataTypeFloat32];
        }
        auto* result = metal::lower(graph, key.op, inputs, p, samples);
        if (!result) {
            std::string types;
            for (int i = 0; i < 4; ++i) {
                const auto input = static_cast<KernelInput>(i);
                if (p.input_rank(input) < 0) continue;
                if (!types.empty()) types += ", ";
                types += dtype_name(DType(p.input_dtype(input)));
            }
            throw std::invalid_argument("Metal does not implement " + key.op + "(" + types + ") -> " +
                                        dtype_name(DType(p.output_dtype())));
        }
        const auto type = metal::storage_type(DType(p.output_dtype()));
        if (type == MPSDataTypeInvalid) throw std::invalid_argument("Metal output dtype unavailable in " + key.op);
        if (result.dataType != type) result = [graph castTensor:result toType:type name:nil];
        result = [graph reshapeTensor:result withShape:metal::storage_shape(p) name:nil];
        MPSGraphCompilationDescriptor* options = [MPSGraphCompilationDescriptor new];
        options.waitForCompilationCompletion = YES;
        executable = [graph compileWithDevice:device feeds:feeds targetTensors:@[result]
                              targetOperations:nil compilationDescriptor:options];
        if (!executable) throw std::runtime_error("Cannot compile Metal operator " + key.op);
        for (MPSGraphTensor* feed in executable.feedTensors) {
            if (feed == samples) {
                slots.at(count++) = -1;
                continue;
            }
            const auto position = std::find(inputs.begin(), inputs.end(), feed);
            if (position == inputs.end() || count >= slots.size())
                throw std::runtime_error("Invalid Metal graph feed mapping");
            slots[count++] = int(position - inputs.begin());
        }
    }
};

void wait_command(id<MTLCommandBuffer> command) {
    if (!command) return;
    [command waitUntilCompleted];
    if (command.status == MTLCommandBufferStatusError) {
        const char* reason = command.error.localizedDescription.UTF8String;
        throw std::runtime_error(std::string("Metal command failed: ") + (reason ? reason : "unknown error"));
    }
}

class MetalContext {
    using Cache = std::list<std::pair<Key, Plan>>;
    Cache cached_;
    std::unordered_map<Key, Cache::iterator, KeyHash> plans_;
    MPSGraphDevice* graph_device_;
    id<MTLCommandQueue> queue_;
    MPSCommandBuffer* pending_ = nil;
    id<MTLCommandBuffer> last_ = nil;
    std::deque<id<MTLCommandBuffer>> submitted_;
    int queued_ = 0;

public:
    id<MTLDevice> device;
    std::mutex mutex;

    MetalContext() {
        @autoreleasepool {
            device = MTLCreateSystemDefaultDevice();
            if (device) {
                queue_ = [device newCommandQueue];
                graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device];
            }
        }
    }

    MPSCommandBuffer* buffer() {
        if (!queue_) throw std::runtime_error("Metal command queue unavailable");
        if (!pending_) pending_ = [MPSCommandBuffer commandBufferFromCommandQueue:queue_];
        return pending_;
    }

    id<MTLCommandBuffer> submit() {
        if (pending_) {
            last_ = pending_.rootCommandBuffer;
            [pending_ commit];
            submitted_.push_back(last_);
            pending_ = nil;
            queued_ = 0;
        }
        while (!submitted_.empty() &&
               (submitted_.size() > 16 || submitted_.front().status >= MTLCommandBufferStatusCompleted)) {
            id<MTLCommandBuffer> previous = submitted_.front();
            submitted_.pop_front();
            if (previous == last_ && previous.status == MTLCommandBufferStatusError) last_ = nil;
            wait_command(previous);
        }
        return last_;
    }

    void synchronize() {
        [submit() waitUntilCompleted];
        // Observe failures from earlier submissions on the same queue.
        submit();
    }

    void encoded() {
        // Bound the live command buffer and submit useful work during long
        // device-only graphs. Host access flushes any partially filled batch.
        const auto& binding = detail::stream_binding(Backend::Metal);
        ++queued_;
        if (binding.stream && !binding.asynchronous) synchronize();
        else if (queued_ >= 32) submit();
    }

    bool ready() const { return !pending_ && (!last_ || last_.status >= MTLCommandBufferStatusCompleted); }

    const Plan& plan(const std::string& op, const KernelParams& p, bool has_samples) {
        Key key{op, p};
        if (random_operation(op)) {
            key.params.v[KernelParams::kAttributeSlot + operator_attr::random::kSeed] = 0;
            key.params.v[KernelParams::kAttributeSlot + operator_attr::random::kStreamOffset] = 0;
        }
        auto existing = plans_.find(key);
        if (existing != plans_.end()) {
            cached_.splice(cached_.begin(), cached_, existing->second);
            return existing->second->second;
        }
        cached_.emplace_front(std::piecewise_construct, std::forward_as_tuple(key),
                             std::forward_as_tuple(graph_device_, key, has_samples));
        try { plans_.emplace(key, cached_.begin()); }
        catch (...) { cached_.pop_front(); throw; }
        constexpr size_t limit = 128;
        if (plans_.size() > limit) {
            plans_.erase(cached_.back().first);
            cached_.pop_back();
        }
        return cached_.front().second;
    }
};

MetalContext& context() { static MetalContext instance; return instance; }

void validate_shape(const BufferPtr& buffer, int rank, const KernelParams& p, int slot) {
    const auto id = static_cast<KernelInput>(slot);
    size_t count = 1;
    for (int dim = rank-1; dim >= 0; --dim) {
        const int extent = slot < 0 ? p.output_shape(dim) : p.input_shape(id,dim);
        if (extent < 0 || (slot >= 0 && size_t(p.input_stride(id,dim)) != count))
            throw std::invalid_argument("Metal requires contiguous tensor descriptors");
        count = checked_memory_mul(count,size_t(extent));
    }
    if (count != buffer->count) throw std::invalid_argument("Metal tensor descriptor size mismatch");
}

MPSGraphTensorData* tensor_data(const BufferPtr& buffer, MPSShape* dims) {
    const auto& owner = storage_for(buffer);
    const auto type = metal::storage_type(buffer->dtype);
    const auto offset = offset_for(buffer, owner);
    if (!offset) return [[MPSGraphTensorData alloc] initWithMTLBuffer:owner.storage shape:dims dataType:type];
    auto* descriptor = [MPSNDArrayDescriptor descriptorWithDataType:type shape:dims];
    MPSNDArray* array = [[MPSNDArray alloc] initWithBuffer:owner.storage offset:offset descriptor:descriptor];
    return [[MPSGraphTensorData alloc] initWithMPSNDArray:array];
}

struct MetalEvent final : detail::EventState {
    id<MTLCommandBuffer> command;
    explicit MetalEvent(id<MTLCommandBuffer> preceding) : EventState(Backend::Metal, 0), command(preceding) {}
    bool ready() const override { return !command || command.status >= MTLCommandBufferStatusCompleted; }
    void synchronize() const override { @autoreleasepool { wait_command(command); } }
};
struct MetalStream final : detail::StreamState {
    MetalStream() : StreamState(Backend::Metal, 0) {}
    std::shared_ptr<detail::EventState> record() override {
        @autoreleasepool {
            auto& state = context();
            std::lock_guard<std::mutex> lock(state.mutex);
            return std::make_shared<MetalEvent>(state.submit());
        }
    }
    void wait(const std::shared_ptr<detail::EventState>&) override {
        // All Metal submissions use the same ordered command queue.
    }
    bool ready() const override {
        auto& state = context();
        std::lock_guard<std::mutex> lock(state.mutex);
        return state.ready();
    }
    void synchronize() const override {
        @autoreleasepool {
            auto& state = context();
            std::lock_guard<std::mutex> lock(state.mutex);
            state.synchronize();
        }
    }
};
} // namespace

bool available_metal() {
    auto device = context().device;
    return device && device.hasUnifiedMemory && [device supportsFamily:MTLGPUFamilyApple7];
}
int allocation_device_metal() {
    if (!available_metal()) throw std::runtime_error("Apple unified-memory Metal device unavailable");
    return 0;
}
std::vector<DeviceInfo> devices_metal() {
    @autoreleasepool {
        if (!available_metal()) return {};
        const auto device = context().device;
        return {{Backend::Metal, 0, device.name.UTF8String, device.recommendedMaxWorkingSetSize}};
    }
}
MemoryInfo memory_info_metal() {
    allocation_device_metal();
    const auto device = context().device;
    const size_t budget = device.recommendedMaxWorkingSetSize, used = device.currentAllocatedSize;
    return {budget, budget > used ? budget - used : 0};
}
BufferPtr allocate_metal(size_t count, DType dtype) {
    allocation_device_metal();
    @autoreleasepool { return std::make_shared<MetalBuffer>(context().device, count, dtype); }
}
void clear_metal(const BufferPtr& buffer) {
    if (!buffer->bytes) return;
    @autoreleasepool {
        auto& state = context();
        std::lock_guard<std::mutex> lock(state.mutex);
        const auto& owner = storage_for(buffer);
        auto encoder = [state.buffer() blitCommandEncoder];
        [encoder fillBuffer:owner.storage range:NSMakeRange(offset_for(buffer, owner), buffer->bytes) value:0];
        [encoder endEncoding];
        state.encoded();
    }
}
void upload_raw_metal(const BufferPtr& buffer, const std::vector<uint8_t>& values) {
    storage_for(buffer);
    if (values.empty()) return;
    @autoreleasepool {
        auto& state = context();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.synchronize();
        std::memcpy(buffer->raw_pointer(), values.data(), values.size());
    }
}
std::vector<uint8_t> download_raw_metal(const BufferPtr& buffer) {
    storage_for(buffer);
    std::vector<uint8_t> bytes(buffer->bytes);
    if (bytes.empty()) return bytes;
    @autoreleasepool {
        auto& state = context();
        std::lock_guard<std::mutex> lock(state.mutex);
        state.synchronize();
        std::memcpy(bytes.data(), buffer->raw_pointer(), bytes.size());
    }
    return bytes;
}
void upload_metal(const BufferPtr& buffer, const std::vector<float>& values) {
    upload_raw_metal(buffer, encode_host_values(values, buffer->dtype));
}
std::vector<float> download_metal(const BufferPtr& buffer) {
    return decode_host_values(download_raw_metal(buffer), buffer->dtype, buffer->count);
}
void upload_int_metal(const BufferPtr& buffer, const std::vector<int32_t>& values) {
    upload_raw_metal(buffer, encode_host_integers(values, buffer->dtype));
}
std::vector<int32_t> download_int_metal(const BufferPtr& buffer) {
    return decode_host_integers(download_raw_metal(buffer), buffer->dtype, buffer->count);
}
void dispatch_metal(const std::string& op, const BufferPtr& output,
                    const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    storage_for(output);
    if (p.output_count() < 0 || size_t(p.output_count()) != output->count || inputs.size() > 4 ||
        p.output_rank() < 0 || p.output_rank() > 8 || p.output_dtype() != int(output->dtype))
        throw std::invalid_argument("Metal dispatch shape or input count mismatch");
    validate_shape(output,p.output_rank(),p,-1);
    for (size_t slot = 0; slot < 4; ++slot) {
        const auto id = static_cast<KernelInput>(slot);
        const auto present = slot < inputs.size() && bool(inputs[slot]);
        if (present != (p.input_rank(id) >= 0) || p.input_rank(id) > 8)
            throw std::invalid_argument("Metal input descriptor does not match the operand");
        if (present) {
            storage_for(inputs[slot]);
            if (p.input_dtype(id) != int(inputs[slot]->dtype))
                throw std::invalid_argument("Metal input dtype mismatch");
            validate_shape(inputs[slot],p.input_rank(id),p,int(slot));
        }
    }
    if (!output->bytes) return;
    @autoreleasepool {
        auto distribution = random_operation(op) ? distribution_for(op,p) : nil;
        const bool direct = distribution && output->dtype == DType::Float32 && op != "dropout";
        BufferPtr samples;
        if (distribution && !direct)
            samples = backend_workspace(Backend::Metal,checked_memory_mul(output->count,sizeof(float)));
        auto& state = context();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (distribution) {
            const auto& target = direct ? output : samples;
            const auto& storage = storage_for(target);
            auto vector = [MPSVectorDescriptor vectorDescriptorWithLength:output->count dataType:MPSDataTypeFloat32];
            auto dest = [[MPSVector alloc] initWithBuffer:storage.storage offset:offset_for(target,storage) descriptor:vector];
            const uint64_t seed = uint32_t(p.attribute(operator_attr::random::kSeed)) |
                (uint64_t(uint32_t(p.attribute(operator_attr::random::kStreamOffset))) << 32);
            auto draw = [[MPSMatrixRandomPhilox alloc] initWithDevice:state.device destinationDataType:MPSDataTypeFloat32
                                                               seed:seed distributionDescriptor:distribution];
            if (!dest || !draw) throw AllocationFailure("Cannot create Metal random generator or destination");
            [draw encodeToCommandBuffer:state.buffer() destinationVector:dest];
            if (direct) {
                state.encoded();
                return;
            }
        }
        if ((op == "copy" || op == "reshape" || op == "squeeze" || op == "unsqueeze") &&
            !inputs.empty() && inputs[0]->dtype == output->dtype && inputs[0]->bytes == output->bytes) {
            const auto& from = storage_for(inputs[0]);
            const auto& to = storage_for(output);
            const size_t source = offset_for(inputs[0], from), destination = offset_for(output, to);
            if (from.storage == to.storage && source == destination) return;
            const auto separation = source < destination ? destination-source : source-destination;
            const bool overlap = from.storage == to.storage && separation < output->bytes;
            id<MTLBuffer> scratch = overlap
                ? [state.device newBufferWithLength:output->bytes options:MTLResourceStorageModePrivate] : nil;
            if (overlap && !scratch) throw AllocationFailure("Cannot allocate an overlapping Metal copy");
            auto encoder = [state.buffer() blitCommandEncoder];
            if (scratch) {
                [encoder copyFromBuffer:from.storage sourceOffset:source toBuffer:scratch destinationOffset:0 size:output->bytes];
                [encoder copyFromBuffer:scratch sourceOffset:0 toBuffer:to.storage destinationOffset:destination size:output->bytes];
            } else {
                [encoder copyFromBuffer:from.storage sourceOffset:source toBuffer:to.storage
                     destinationOffset:destination size:output->bytes];
            }
            [encoder endEncoding];
        } else {
            const auto& plan = state.plan(op, p, samples != nullptr);
            NSMutableArray<MPSGraphTensorData*>* feeds = [NSMutableArray arrayWithCapacity:plan.count];
            for (size_t i = 0; i < plan.count; ++i) {
                const int slot = plan.slots[i];
                if (slot < 0)
                    [feeds addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:storage_for(samples).storage
                        shape:metal::shape(p) dataType:MPSDataTypeFloat32]];
                else [feeds addObject:tensor_data(inputs[slot], metal::storage_shape(p, slot))];
            }
            [plan.executable encodeToCommandBuffer:state.buffer() inputsArray:feeds
                                     resultsArray:@[tensor_data(output, metal::storage_shape(p))] executionDescriptor:nil];
        }
        state.encoded();
    }
}
std::shared_ptr<detail::StreamState> detail::create_metal_stream() {
    allocation_device_metal();
    return std::make_shared<MetalStream>();
}
} // namespace elman
