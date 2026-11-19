#pragma once

#include "Core/Backend.h"
#include <limits>
#include <stdexcept>

namespace elman {
enum class MemoryKind { Tensor, Workspace, Cached };
struct MemoryStats {
    size_t tensor_bytes = 0, workspace_bytes = 0, cached_bytes = 0, peak_bytes = 0;
    uint64_t allocations = 0; // Accepted reservations, including subsequent allocator failures.
    size_t live_bytes() const { return tensor_bytes + workspace_bytes + cached_bytes; }
};
struct MemoryInfo { size_t total_bytes = 0, available_bytes = 0; };
class MemoryLimitExceeded : public std::runtime_error {
public:
    MemoryLimitExceeded(Backend backend, size_t requested, size_t live, size_t limit);
};
class AllocationFailure : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
// Payload and registered workspace accounting, independent of vendor/OS caches.
MemoryStats memory_stats(Backend backend);
MemoryStats memory_stats(Backend backend, int device);
MemoryInfo memory_info(Backend backend);
size_t memory_allocation_bytes(Backend backend, size_t payload);
size_t memory_available_limit(Backend backend, size_t maximum, size_t reserve);
size_t checked_memory_add(size_t a, size_t b);
size_t checked_memory_mul(size_t a, size_t b);

class MemoryReservation {
public:
    MemoryReservation() = default;
    MemoryReservation(Backend backend, size_t bytes, MemoryKind kind = MemoryKind::Workspace);
    ~MemoryReservation();
    MemoryReservation(const MemoryReservation&) = delete;
    MemoryReservation& operator=(const MemoryReservation&) = delete;
    MemoryReservation(MemoryReservation&& other) noexcept;
    MemoryReservation& operator=(MemoryReservation&& other) noexcept;
    void reclassify(MemoryKind kind);
private:
    void release() noexcept;
    Backend backend_ = Backend::CPU;
    int device_ = -1;
    MemoryKind kind_ = MemoryKind::Workspace;
    size_t bytes_ = 0;
};

// Limits apply process-wide to one backend/device. Nested/concurrent scopes enforce the
// smallest active ceiling; reservations and updates are atomic under a mutex.
class MemoryLimit {
public:
    MemoryLimit(Backend backend, size_t bytes);
    ~MemoryLimit();
    MemoryLimit(const MemoryLimit&) = delete;
    MemoryLimit& operator=(const MemoryLimit&) = delete;
    size_t peak_bytes() const;
private:
    Backend backend_;
    int device_;
    uint64_t id_;
};
} // namespace elman
