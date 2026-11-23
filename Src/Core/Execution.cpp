#include "Core/Execution.h"
#include "Core/Memory.h"
#ifdef ELMAN_HAS_METAL
#include "Backends/Metal.h"
#endif
#include <array>
#include <stdexcept>

namespace elman {
namespace detail {
#ifdef ELMAN_HAS_CUDA
std::shared_ptr<StreamState> create_cuda_stream(const HostStagingOptions& options);
#endif
StreamBinding& stream_binding(Backend backend) {
    const size_t index = static_cast<size_t>(backend);
    thread_local std::array<StreamBinding, 3> bindings;
    if (index >= bindings.size()) throw std::invalid_argument("Invalid execution backend");
    return bindings[index];
}
}
namespace {
struct CpuEvent final : detail::EventState {
    CpuEvent() : EventState(Backend::CPU, -1) {}
    bool ready() const override { return true; }
    void synchronize() const override {}
};
struct CpuStream final : detail::StreamState {
    CpuStream() : StreamState(Backend::CPU, -1) {}
    std::shared_ptr<detail::EventState> record() override { return std::make_shared<CpuEvent>(); }
    void wait(const std::shared_ptr<detail::EventState>& event) override { event->synchronize(); }
    bool ready() const override { return true; }
    void synchronize() const override {}
};
}
size_t host_staging_memory_bound(const HostStagingOptions& options) {
    if (!options.capacity_bytes || !options.max_buffers) return 0;
    return checked_memory_add(options.capacity_bytes, checked_memory_mul(options.max_buffers, size_t(128)));
}
ExecutionStream::ExecutionStream(Backend backend, const HostStagingOptions& staging) {
    host_staging_memory_bound(staging);
    if (backend == Backend::CPU) state_ = std::make_shared<CpuStream>();
#ifdef ELMAN_HAS_CUDA
    else if (backend == Backend::CUDA) state_ = detail::create_cuda_stream(staging);
#endif
#ifdef ELMAN_HAS_METAL
    else if (backend == Backend::Metal) state_ = detail::create_metal_stream();
#endif
    else throw std::invalid_argument("Execution backend unavailable");
}
ExecutionEvent ExecutionStream::record() const { return ExecutionEvent(state_->record()); }
void ExecutionStream::wait(const ExecutionEvent& event) const {
    if (!event.state_ || event.state_->backend != backend() || event.state_->device != device())
        throw std::invalid_argument("Event and stream must belong to the same backend and device");
    state_->wait(event.state_);
}
bool ExecutionStream::ready() const { return state_->ready(); }
void ExecutionStream::synchronize() const { state_->synchronize(); }
HostStagingStats ExecutionStream::host_staging_stats() const { return state_->host_staging_stats(); }
void ExecutionStream::trim_host_staging() const { state_->trim_host_staging(); }
bool ExecutionEvent::ready() const {
    if (!state_) throw std::invalid_argument("Unrecorded execution event");
    return state_->ready();
}
void ExecutionEvent::synchronize() const {
    if (!state_) throw std::invalid_argument("Unrecorded execution event");
    state_->synchronize();
}
StreamScope::StreamScope(const ExecutionStream& stream, bool asynchronous, bool pinned_staging)
    : state_(stream.state_), previous_(detail::stream_binding(stream.backend())), previous_device_(state_->enter()) {
    detail::stream_binding(stream.backend()) = {state_, asynchronous, pinned_staging};
}
StreamScope::~StreamScope() {
    detail::stream_binding(state_->backend) = std::move(previous_);
    state_->leave(previous_device_);
}
bool asynchronous_execution(Backend backend) {
    return backend != Backend::CPU && detail::stream_binding(backend).asynchronous;
}
bool pinned_staging_enabled(Backend backend) {
    return backend == Backend::CUDA && asynchronous_execution(backend) && detail::stream_binding(backend).pinned_staging;
}
} // namespace elman
