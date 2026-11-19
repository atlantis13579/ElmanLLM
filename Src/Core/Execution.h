#pragma once
#include "Core/Backend.h"
#include <utility>

namespace elman {
struct HostStagingOptions {
    size_t capacity_bytes = 4 * 1024 * 1024;
    size_t max_buffers = 256;
};
struct HostStagingStats {
    size_t allocated_bytes = 0, pending_bytes = 0, cached_bytes = 0, peak_bytes = 0;
    size_t buffers = 0, metadata_bytes = 0;
    uint64_t allocations = 0, reuses = 0, waits = 0, fallbacks = 0;
    uint64_t uploads = 0, downloads = 0, uploaded_bytes = 0, downloaded_bytes = 0;
};
// Includes bounded payload and descriptor storage; vendor/OS overhead is excluded.
size_t host_staging_memory_bound(const HostStagingOptions& options = {});
namespace detail {
struct EventState {
    Backend backend;
    int device;
    EventState(Backend backend_, int device_) : backend(backend_), device(device_) {}
    virtual ~EventState() = default;
    virtual bool ready() const = 0;
    virtual void synchronize() const = 0;
};
struct StreamState {
    Backend backend;
    int device;
    StreamState(Backend backend_, int device_) : backend(backend_), device(device_) {}
    virtual ~StreamState() = default;
    virtual std::shared_ptr<EventState> record() = 0;
    virtual void wait(const std::shared_ptr<EventState>& event) = 0;
    virtual bool ready() const = 0;
    virtual void synchronize() const = 0;
    virtual int enter() const { return -1; }
    virtual void leave(int) const noexcept {}
    virtual HostStagingStats host_staging_stats() const { return {}; }
    virtual void trim_host_staging() const {}
};
struct StreamBinding {
    std::shared_ptr<StreamState> stream;
    bool asynchronous = false, pinned_staging = false;
};
StreamBinding& stream_binding(Backend backend);
}

class ExecutionEvent {
public:
    ExecutionEvent() = default;
    bool ready() const;
    void synchronize() const;
private:
    explicit ExecutionEvent(std::shared_ptr<detail::EventState> state) : state_(std::move(state)) {}
    std::shared_ptr<detail::EventState> state_;
    friend class ExecutionStream;
};

// CPU streams execute immediately. CUDA streams are independent nonblocking
// queues. Events are immutable snapshots, and host downloads always wait.
class ExecutionStream {
public:
    explicit ExecutionStream(Backend backend, const HostStagingOptions& staging = {});
    Backend backend() const { return state_->backend; }
    int device() const { return state_->device; }
    ExecutionEvent record() const;
    void wait(const ExecutionEvent& event) const;
    bool ready() const;
    void synchronize() const;
    HostStagingStats host_staging_stats() const;
    // Releases completed staging buffers. Pending transfers retain their storage.
    void trim_host_staging() const;
private:
    std::shared_ptr<detail::StreamState> state_;
    friend class StreamScope;
};

// Selects a queue on the calling thread and restores the previous binding and
// device. Scope exit does not wait; explicit barriers report execution errors.
class StreamScope {
public:
    // Pinned staging is explicit: ordinary uploads retain their host barrier.
    explicit StreamScope(const ExecutionStream& stream, bool asynchronous = true, bool pinned_staging = false);
    ~StreamScope();
    StreamScope(const StreamScope&) = delete;
    StreamScope& operator=(const StreamScope&) = delete;
private:
    std::shared_ptr<detail::StreamState> state_;
    detail::StreamBinding previous_;
    int previous_device_;
};
bool asynchronous_execution(Backend backend);
bool pinned_staging_enabled(Backend backend);
} // namespace elman
