#include "Core/backend.h"
#include "Backends/host_words.h"
#include "Backends/shader_source.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace elman {
namespace {
std::string metal_error(NSError* error) {
    return error ? std::string([[error localizedDescription] UTF8String]) : "unknown Metal error";
}

struct MetalPipeline { id<MTLComputePipelineState> value; };

struct MetalState {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    std::unordered_map<std::string, MetalPipeline> pipelines;
    std::mutex mutex;

    MetalState() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) throw std::runtime_error("No Metal device is available");
        queue = [device newCommandQueue];
        if (!queue) throw std::runtime_error("Cannot create a Metal command queue");
    }

    id<MTLComputePipelineState> pipeline(const std::string& op) {
        auto found = pipelines.find(op);
        if (found != pipelines.end()) return found->second.value;
        if (op.empty() || !std::all_of(op.begin(), op.end(), [](unsigned char ch) {
                return (ch >= 'a' && ch <= 'z') || (ch >= '0' && ch <= '9') || ch == '_';
            })) throw std::runtime_error("Invalid Metal operator name");
        const auto path = runtime_shader_path("Metal", op + "_metal.metal");
        const auto source = load_shader_source(path);
        NSString* text = [[NSString alloc] initWithBytes:source.data() length:source.size() encoding:NSUTF8StringEncoding];
        if (!text) throw std::runtime_error("Metal shader source is not valid UTF-8");
        MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
        options.fastMathEnabled = NO;
        NSError* error = nil;
        id<MTLLibrary> library = [device newLibraryWithSource:text options:options error:&error];
        if (!library) throw std::runtime_error("Metal shader compilation failed for " + op + ": " + metal_error(error));
        id<MTLFunction> function = [library newFunctionWithName:@"elman_main"];
        if (!function) throw std::runtime_error("Metal shader has no elman_main function: " + op);
        id<MTLComputePipelineState> compiled = [device newComputePipelineStateWithFunction:function error:&error];
        if (!compiled) throw std::runtime_error("Metal pipeline creation failed for " + op + ": " + metal_error(error));
        pipelines.emplace(op, MetalPipeline{compiled});
        return compiled;
    }
};

MetalState& metal_state() { static MetalState state; return state; }

struct MetalBuffer final : Buffer {
    id<MTLBuffer> data;

    MetalBuffer(std::size_t size, DType type) : Buffer(Backend::Metal, type, size) {
        const std::size_t words = std::max(size, std::size_t(1));
        if (words > std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t))
            throw std::runtime_error("Metal allocation size overflow");
        auto& state = metal_state();
        data = [state.device newBufferWithLength:words * sizeof(std::uint32_t) options:MTLResourceStorageModeShared];
        if (!data) throw std::runtime_error("Metal buffer allocation failed");
        std::memset([data contents], 0, words * sizeof(std::uint32_t));
    }
};

MetalBuffer& checked_metal(const BufferPtr& buffer) {
    auto* value = dynamic_cast<MetalBuffer*>(buffer.get());
    if (!value) throw std::runtime_error("Expected a Metal buffer");
    return *value;
}
}

bool available_metal() {
    @autoreleasepool { try { metal_state(); return true; } catch (...) { return false; } }
}

BufferPtr allocate_metal(std::size_t count, DType dtype) {
    @autoreleasepool { return std::make_shared<MetalBuffer>(count, dtype); }
}

void upload_metal(const BufferPtr& buffer, const std::vector<float>& values) {
    @autoreleasepool {
        auto& target = checked_metal(buffer);
        if (values.size() != target.count) throw std::runtime_error("Metal upload element count mismatch");
        const auto words = encode_words(values, target.dtype);
        auto& state = metal_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        if (!words.empty()) std::memcpy([target.data contents], words.data(), words.size() * sizeof(std::uint32_t));
    }
}

std::vector<float> download_metal(const BufferPtr& buffer) {
    @autoreleasepool {
        auto& source = checked_metal(buffer);
        auto& state = metal_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        std::vector<std::uint32_t> words(source.count);
        if (!words.empty()) std::memcpy(words.data(), [source.data contents], words.size() * sizeof(std::uint32_t));
        return decode_words(words, source.dtype);
    }
}

void dispatch_metal(const std::string& op, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& params) {
    @autoreleasepool {
        auto& target = checked_metal(output);
        if (inputs.size() > 3) throw std::runtime_error("Metal kernels accept at most three inputs");
        if (params.v[0] < 0 || static_cast<std::size_t>(params.v[0]) != target.count)
            throw std::runtime_error("Metal dispatch output count mismatch");
        if (target.count == 0) return;
        auto& state = metal_state();
        std::lock_guard<std::mutex> lock(state.mutex);
        id<MTLComputePipelineState> pipeline = state.pipeline(op);
        id<MTLCommandBuffer> command = [state.queue commandBuffer];
        if (!command) throw std::runtime_error("Cannot create Metal command buffer");
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!encoder) throw std::runtime_error("Cannot create Metal compute encoder");
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:target.data offset:0 atIndex:0];
        for (std::size_t i = 0; i < 3; ++i) {
            id<MTLBuffer> source = i < inputs.size() && inputs[i] ? checked_metal(inputs[i]).data : target.data;
            [encoder setBuffer:source offset:0 atIndex:i + 1];
        }
        [encoder setBytes:&params length:sizeof(KernelParams) atIndex:4];
        const NSUInteger width = std::min(NSUInteger(64), [pipeline maxTotalThreadsPerThreadgroup]);
        const MTLSize groups = MTLSizeMake((target.count + width - 1) / width, 1, 1);
        [encoder dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if ([command status] == MTLCommandBufferStatusError)
            throw std::runtime_error("Metal execution failed for " + op + ": " + metal_error([command error]));
    }
}
}
