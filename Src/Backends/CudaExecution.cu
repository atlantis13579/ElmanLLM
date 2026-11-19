#include "Backends/CudaExecution.h"
#include "Core/Memory.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <stdexcept>

namespace elman {
void check_cuda_execution(cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    cudaGetLastError();
    auto message = std::string(operation) + ": " + cudaGetErrorString(status);
    if (status == cudaErrorMemoryAllocation) throw AllocationFailure(message);
    throw std::runtime_error(message);
}
std::recursive_mutex& cuda_submission_mutex() {
    static auto* value = new std::recursive_mutex;
    return *value;
}
namespace {
struct DeviceGuard {
    int previous = -1;
    explicit DeviceGuard(int device) {
        check_cuda_execution(cudaGetDevice(&previous), "cudaGetDevice");
        if (device != previous) check_cuda_execution(cudaSetDevice(device), "cudaSetDevice");
        else previous = -1;
    }
    ~DeviceGuard() { if (previous >= 0) cudaSetDevice(previous); }
};
std::atomic<uint64_t> next_stream{1};
uint64_t default_stream_id(int device) { return (uint64_t(1) << 63) | uint64_t(device); }

// Once the last host reference is gone, an already submitted stream wait
// retains its recorded state. CUDA permits the event to be recorded again.
class CudaEventPool {
    int device_;
    std::mutex mutex_;
    std::array<cudaEvent_t, 32> idle_{};
    size_t count_ = 0;
public:
    explicit CudaEventPool(int device) : device_(device) {}
    ~CudaEventPool() {
        int previous = -1;
        cudaGetDevice(&previous);
        if (previous != device_) cudaSetDevice(device_);
        for (size_t i = 0; i < count_; ++i) cudaEventDestroy(idle_[i]);
        if (previous >= 0 && previous != device_) cudaSetDevice(previous);
    }
    cudaEvent_t take() {
        std::lock_guard<std::mutex> lock(mutex_);
        return count_ ? idle_[--count_] : nullptr;
    }
    bool put(cudaEvent_t event) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (count_ == idle_.size()) return false;
        idle_[count_++] = event;
        return true;
    }
};
}
struct CudaEventState final : detail::EventState {
    cudaEvent_t event = nullptr;
    uint64_t stream_id;
    std::shared_ptr<CudaEventPool> pool;
    CudaEventState(int device_, uint64_t id, cudaStream_t stream, std::shared_ptr<CudaEventPool> pool_ = {})
        : EventState(Backend::CUDA, device_), stream_id(id), pool(std::move(pool_)) {
        if (pool) event = pool->take();
        if (!event) check_cuda_execution(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), "cudaEventCreate");
        try { check_cuda_execution(cudaEventRecord(event, stream), "cudaEventRecord"); }
        catch (...) { cudaEventDestroy(event); throw; }
    }
    ~CudaEventState() override {
        if (pool && pool->put(event)) return;
        int previous = -1;
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
        cudaEventDestroy(event);
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
    bool ready() const override {
        DeviceGuard guard(device);
        auto status = cudaEventQuery(event);
        if (status == cudaErrorNotReady) { cudaGetLastError(); return false; }
        check_cuda_execution(status, "cudaEventQuery");
        return true;
    }
    void synchronize() const override {
        DeviceGuard guard(device);
        check_cuda_execution(cudaEventSynchronize(event), "cudaEventSynchronize");
    }
};
namespace {
// The stream, rather than a tensor's latest writer, owns pending host storage.
// Replacing a writer event must never release an earlier upload's source bytes.
class HostStagingPool {
    struct Slot {
        void* data = nullptr;
        size_t bytes = 0;
        MemoryReservation reservation;
        std::shared_ptr<CudaEventState> completion;
        bool busy = false;
    };
    static_assert(sizeof(Slot) <= 128, "Host staging descriptor bound is too small");
    HostStagingOptions options_;
    HostStagingStats stats_;
    MemoryReservation metadata_;
    std::vector<Slot> slots_;

    void recycle(Slot& slot) {
        slot.completion.reset();
        slot.busy = false;
        slot.reservation.reclassify(MemoryKind::Cached);
    }
    void remove(size_t index) {
        check_cuda_execution(cudaFreeHost(slots_[index].data), "cudaFreeHost");
        stats_.allocated_bytes -= slots_[index].bytes;
        slots_.erase(slots_.begin() + index);
    }
    Slot* acquire_impl(size_t bytes) {
        reap();
        for (;;) {
            Slot* best = nullptr;
            for (auto& slot : slots_) if (!slot.busy && slot.bytes >= bytes && (!best || slot.bytes < best->bytes)) best = &slot;
            if (best) {
                best->reservation.reclassify(MemoryKind::Workspace);
                best->busy = true;
                ++stats_.reuses;
                return best;
            }
            for (size_t i = slots_.size(); i > 0; --i) if (!slots_[i - 1].busy) remove(i - 1);
            if (slots_.size() < options_.max_buffers && bytes <= options_.capacity_bytes - stats_.allocated_bytes) break;
            auto pending = std::find_if(slots_.begin(), slots_.end(), [](const Slot& slot) { return bool(slot.completion); });
            if (pending == slots_.end()) throw std::logic_error("Host staging capacity has no pending transfer");
            ++stats_.waits;
            pending->completion->synchronize();
            reap();
        }
        if (!slots_.capacity()) {
            MemoryReservation metadata(Backend::CPU, checked_memory_mul(options_.max_buffers, size_t(128)), MemoryKind::Cached);
            slots_.reserve(options_.max_buffers);
            metadata_ = std::move(metadata);
            stats_.metadata_bytes = options_.max_buffers * 128;
        }
        Slot slot;
        slot.bytes = bytes;
        slot.reservation = MemoryReservation(Backend::CPU, bytes);
        check_cuda_execution(cudaHostAlloc(&slot.data, bytes, cudaHostAllocDefault), "cudaHostAlloc");
        slot.busy = true;
        slots_.push_back(std::move(slot));
        stats_.allocated_bytes += bytes;
        stats_.peak_bytes = std::max(stats_.peak_bytes, stats_.allocated_bytes);
        ++stats_.allocations;
        return &slots_.back();
    }
    Slot* acquire(size_t bytes) {
        if (!bytes || !options_.max_buffers || bytes > options_.capacity_bytes || options_.capacity_bytes < 4096) {
            ++stats_.fallbacks;
            return nullptr;
        }
        size_t rounded = 4096;
        while (rounded < bytes && rounded <= options_.capacity_bytes / 2) rounded *= 2;
        if (rounded < bytes) {
            size_t padding = (4096 - bytes % 4096) % 4096;
            if (padding > options_.capacity_bytes - bytes) {
                ++stats_.fallbacks;
                return nullptr;
            }
            rounded = bytes + padding;
        }
        try { return acquire_impl(rounded); }
        catch (const MemoryLimitExceeded&) {}
        catch (const AllocationFailure&) {}
        catch (const std::bad_alloc&) {}
        ++stats_.fallbacks;
        return nullptr;
    }
public:
    explicit HostStagingPool(const HostStagingOptions& options) : options_(options) {}
    void reap() {
        for (auto& slot : slots_) if (slot.completion && slot.completion->ready()) recycle(slot);
    }
    void trim() {
        reap();
        for (size_t i = slots_.size(); i > 0; --i) if (!slots_[i - 1].busy) remove(i - 1);
        if (slots_.empty()) {
            std::vector<Slot>().swap(slots_);
            metadata_ = {};
            stats_.metadata_bytes = 0;
        }
    }
    // The owner has already synchronized its queue, including on exceptional exit.
    void clear() noexcept {
        for (auto& slot : slots_) cudaFreeHost(slot.data);
        slots_.clear();
        std::vector<Slot>().swap(slots_);
        metadata_ = {};
        stats_.allocated_bytes = stats_.metadata_bytes = 0;
    }
    HostStagingStats stats() {
        reap();
        auto result = stats_;
        result.buffers = slots_.size();
        for (const auto& slot : slots_) {
            if (slot.busy) result.pending_bytes += slot.bytes;
            else result.cached_bytes += slot.bytes;
        }
        return result;
    }
    std::shared_ptr<CudaEventState> upload(void* destination, const void* source, size_t bytes) {
        auto* slot = acquire(bytes);
        if (!slot) return {};
        std::memcpy(slot->data, source, bytes);
        try {
            check_cuda_execution(cudaMemcpyAsync(destination, slot->data, bytes, cudaMemcpyHostToDevice, current_cuda_stream()), "Staged CUDA upload");
            slot->completion = record_cuda_completion();
            ++stats_.uploads;
            stats_.uploaded_bytes += bytes;
            return slot->completion;
        } catch (...) {
            cudaStreamSynchronize(current_cuda_stream());
            recycle(*slot);
            throw;
        }
    }
    bool download(void* destination, const void* source, size_t bytes) {
        auto* slot = acquire(bytes);
        if (!slot) return false;
        try {
            check_cuda_execution(cudaMemcpyAsync(slot->data, source, bytes, cudaMemcpyDeviceToHost, current_cuda_stream()), "Staged CUDA download");
            check_cuda_execution(cudaStreamSynchronize(current_cuda_stream()), "Staged CUDA download completion");
            std::memcpy(destination, slot->data, bytes);
            recycle(*slot);
            ++stats_.downloads;
            stats_.downloaded_bytes += bytes;
            return true;
        } catch (...) {
            cudaStreamSynchronize(current_cuda_stream());
            recycle(*slot);
            throw;
        }
    }
};
struct CudaStreamState final : detail::StreamState {
    cudaStream_t stream = nullptr;
    uint64_t id = next_stream++;
    std::shared_ptr<void> blas;
    std::shared_ptr<CudaEventPool> events;
    mutable HostStagingPool staging;
    explicit CudaStreamState(const HostStagingOptions& options) : StreamState(Backend::CUDA, 0), staging(options) {
        check_cuda_execution(cudaGetDevice(&device), "cudaGetDevice");
        events = std::make_shared<CudaEventPool>(device);
        check_cuda_execution(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "cudaStreamCreate");
    }
    ~CudaStreamState() override {
        int previous = -1;
        cudaGetDevice(&previous);
        if (previous != device) cudaSetDevice(device);
        cudaStreamSynchronize(stream);
        staging.clear();
        blas.reset();
        cudaStreamDestroy(stream);
        if (previous >= 0 && previous != device) cudaSetDevice(previous);
    }
    std::shared_ptr<detail::EventState> record() override {
        std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
        DeviceGuard guard(device);
        return std::make_shared<CudaEventState>(device, id, stream, events);
    }
    void wait(const std::shared_ptr<detail::EventState>& event) override {
        std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
        DeviceGuard guard(device);
        auto native = std::dynamic_pointer_cast<CudaEventState>(event);
        if (!native) throw std::invalid_argument("Expected a CUDA event");
        check_cuda_execution(cudaStreamWaitEvent(stream, native->event, 0), "cudaStreamWaitEvent");
    }
    bool ready() const override {
        DeviceGuard guard(device);
        auto status = cudaStreamQuery(stream);
        if (status == cudaErrorNotReady) { cudaGetLastError(); return false; }
        check_cuda_execution(status, "cudaStreamQuery");
        return true;
    }
    void synchronize() const override {
        DeviceGuard guard(device);
        check_cuda_execution(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
        staging.reap();
    }
    HostStagingStats host_staging_stats() const override {
        std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
        DeviceGuard guard(device);
        return staging.stats();
    }
    void trim_host_staging() const override {
        std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
        DeviceGuard guard(device);
        staging.trim();
    }
    int enter() const override {
        int previous = -1;
        check_cuda_execution(cudaGetDevice(&previous), "cudaGetDevice");
        if (previous != device) check_cuda_execution(cudaSetDevice(device), "cudaSetDevice");
        return previous;
    }
    void leave(int previous) const noexcept override { if (previous >= 0 && previous != device) cudaSetDevice(previous); }
};
CudaStreamState* current() {
    return static_cast<CudaStreamState*>(detail::stream_binding(Backend::CUDA).stream.get());
}
void wait_for(const std::shared_ptr<CudaEventState>& event) {
    if (!event || event->stream_id == current_cuda_stream_id()) return;
    check_cuda_execution(cudaStreamWaitEvent(current_cuda_stream(), event->event, 0), "CUDA buffer dependency");
}
}
namespace detail {
std::shared_ptr<StreamState> create_cuda_stream(const HostStagingOptions& options) { return std::make_shared<CudaStreamState>(options); }
}
cudaStream_t current_cuda_stream() { auto* value = current(); return value ? value->stream : nullptr; }
uint64_t current_cuda_stream_id() {
    if (auto* value = current()) return value->id;
    int device = 0;
    check_cuda_execution(cudaGetDevice(&device), "Default stream device");
    return default_stream_id(device);
}
std::shared_ptr<void>& current_cuda_blas() {
    if (auto* value = current()) return value->blas;
    int device = 0;
    check_cuda_execution(cudaGetDevice(&device), "cudaGetDevice");
    thread_local std::map<int, std::shared_ptr<void>> contexts;
    return contexts[device];
}
void synchronize_cuda_stream() { check_cuda_execution(cudaStreamSynchronize(current_cuda_stream()), "CUDA stream execution"); }
std::shared_ptr<CudaEventState> stage_cuda_upload(void* destination, const void* source, size_t bytes) {
    auto* stream = current();
    return stream && pinned_staging_enabled(Backend::CUDA) ? stream->staging.upload(destination, source, bytes) : nullptr;
}
bool stage_cuda_download(void* destination, const void* source, size_t bytes) {
    auto* stream = current();
    return stream && pinned_staging_enabled(Backend::CUDA) && stream->staging.download(destination, source, bytes);
}
std::shared_ptr<CudaEventState> record_cuda_completion() {
    if (auto* state = current())
        return std::make_shared<CudaEventState>(state->device, state->id, state->stream, state->events);
    int device = 0;
    check_cuda_execution(cudaGetDevice(&device), "cudaGetDevice");
    thread_local std::map<int, std::shared_ptr<CudaEventPool>> pools;
    auto& pool = pools[device];
    if (!pool) pool = std::make_shared<CudaEventPool>(device);
    return std::make_shared<CudaEventState>(device, default_stream_id(device), nullptr, pool);
}
void prepare_cuda_read(const CudaAccess& access) { wait_for(access.writer); }
void prepare_cuda_write(const CudaAccess& access) {
    wait_for(access.writer);
    for (const auto& reader : access.readers) wait_for(reader.second);
}
void finish_cuda_read(CudaAccess& access, const std::shared_ptr<CudaEventState>& event) {
    if (!event) return;
    // Retire old queues so persistent weights do not retain every past request.
    for (auto it = access.readers.begin(); it != access.readers.end();) {
        if (it->first != event->stream_id && it->second->ready()) it = access.readers.erase(it);
        else ++it;
    }
    access.readers[event->stream_id] = event;
}
void finish_cuda_write(CudaAccess& access, const std::shared_ptr<CudaEventState>& event) {
    access.writer = event;
    access.readers.clear();
}
} // namespace elman
