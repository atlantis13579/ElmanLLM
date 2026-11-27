#pragma once

#include "Core/Memory.h"

namespace elman {
struct AllocationPoolState;
struct AllocationPoolStats {
    uint64_t requests = 0, allocations = 0, reuses = 0, evictions = 0;
    size_t allocated_bytes = 0, reserved_bytes = 0, peak_reserved_bytes = 0;
    size_t cached_bytes = 0, peak_cached_bytes = 0;
};

// Thread-local allocation routing. CUDA leases retain access events so reuse
// on another stream waits for earlier readers and writers. Leases
// may outlive a scope or be released by another thread; live buffers are never
// recycled. Idle buffers are charged until released. The optional absolute
// target trims caches, while MemoryLimit remains the hard allocation ceiling.
class AllocationPool {
public:
    explicit AllocationPool(Backend backend, size_t target_bytes = 0, bool reuse = true);
    ~AllocationPool();
    AllocationPool(const AllocationPool&) = delete;
    AllocationPool& operator=(const AllocationPool&) = delete;
    AllocationPoolStats stats() const;
    void clear();
private:
    std::shared_ptr<AllocationPoolState> state_;
};

// Byte-addressed, aligned scratch storage. Contents are unspecified; kernals
// must initialize every value they read. Ownership follows BufferPtr aliases.
BufferPtr backend_workspace(Backend backend, size_t bytes);

namespace detail {
using AllocateBuffer = BufferPtr (*)(size_t, DType);
using ClearBuffer = void (*)(const BufferPtr&);
int allocation_device(Backend backend);
BufferPtr pooled_allocate(Backend backend, size_t count, DType dtype, MemoryKind kind,
                          AllocateBuffer allocate, ClearBuffer clear);
}
} // namespace elman
