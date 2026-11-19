#include "Core/AllocationPool.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <tuple>

namespace elman {
namespace {
struct Allocation {
    MemoryReservation reservation;
    BufferPtr buffer;
    MemoryKind kind;
    size_t bytes;
    Allocation(Backend backend, size_t count, DType dtype, MemoryKind memory_kind, detail::AllocateBuffer allocate)
        : reservation(backend, memory_allocation_bytes(backend, dtype_storage_bytes(dtype, count)), memory_kind),
          buffer(allocate(count, dtype)), kind(memory_kind),
          bytes(memory_allocation_bytes(backend, buffer->bytes)) {}
};
using Key = std::tuple<MemoryKind, DType, size_t>;
thread_local std::vector<std::weak_ptr<AllocationPoolState>> pools;
}

struct AllocationPoolState {
    Backend backend;
    int device;
    size_t target;
    bool reuse, closed = false;
    mutable std::mutex mutex;
    AllocationPoolStats counters;
    std::map<Key, std::vector<std::shared_ptr<Allocation>>> idle;

    AllocationPoolState(Backend backend_, size_t target_, bool reuse_)
        : backend(backend_), device(detail::allocation_device(backend_)), target(target_), reuse(reuse_) {}
    bool active() const {
        std::lock_guard<std::mutex> lock(mutex);
        return !closed;
    }
    AllocationPoolStats stats() const {
        std::lock_guard<std::mutex> lock(mutex);
        return counters;
    }
    std::shared_ptr<Allocation> take(MemoryKind kind, DType dtype, size_t count) {
        std::lock_guard<std::mutex> lock(mutex);
        ++counters.requests;
        auto found = idle.find(Key{kind, dtype, count});
        if (found == idle.end()) return {};
        if (found->second.empty()) { idle.erase(found); return {}; }
        auto allocation = std::move(found->second.back());
        found->second.pop_back();
        if (found->second.empty()) idle.erase(found);
        counters.cached_bytes -= allocation->bytes;
        ++counters.reuses;
        allocation->reservation.reclassify(kind);
        return allocation;
    }
    void added(const std::shared_ptr<Allocation>& allocation) {
        std::lock_guard<std::mutex> lock(mutex);
        ++counters.allocations;
        counters.allocated_bytes += allocation->bytes;
        counters.reserved_bytes += allocation->bytes;
        counters.peak_reserved_bytes = std::max(counters.peak_reserved_bytes, counters.reserved_bytes);
    }
    void release(std::shared_ptr<Allocation> allocation, bool recyclable) noexcept {
        std::lock_guard<std::mutex> lock(mutex);
        if (!closed && reuse && recyclable && (!target || memory_stats(backend, device).live_bytes() <= target)) {
            auto key = Key{allocation->kind, allocation->buffer->dtype, allocation->buffer->count};
            bool retained = false;
            try {
                idle[key].push_back(allocation);
                retained = true;
            } catch (...) {
                // Returning a lease must not throw during exception unwinding.
                auto found = idle.find(key);
                if (found != idle.end() && found->second.empty()) idle.erase(found);
            }
            if (retained) {
                allocation->reservation.reclassify(MemoryKind::Cached);
                counters.cached_bytes += allocation->bytes;
                counters.peak_cached_bytes = std::max(counters.peak_cached_bytes, counters.cached_bytes);
                return;
            }
        }
        counters.reserved_bytes -= allocation->bytes;
    }
    size_t clear(bool closing = false) {
        std::lock_guard<std::mutex> lock(mutex);
        if (closing) closed = true;
        auto released = counters.cached_bytes;
        for (const auto& item : idle) counters.evictions += item.second.size();
        counters.reserved_bytes -= released;
        counters.cached_bytes = 0;
        idle.clear();
        return released;
    }
};

namespace {
struct Lease {
    std::shared_ptr<AllocationPoolState> pool;
    std::shared_ptr<Allocation> allocation;
    bool recyclable = true;
    Lease(std::shared_ptr<AllocationPoolState> pool_, std::shared_ptr<Allocation> allocation_)
        : pool(std::move(pool_)), allocation(std::move(allocation_)) {}
    ~Lease() { pool->release(std::move(allocation), recyclable); }
};
std::shared_ptr<AllocationPoolState> current(Backend backend, int device) {
    for (auto it = pools.rbegin(); it != pools.rend(); ++it) {
        auto pool = it->lock();
        if (pool && pool->backend == backend && pool->device == device && pool->active()) return pool;
    }
    return {};
}
}

AllocationPool::AllocationPool(Backend backend, size_t target_bytes, bool reuse)
    : state_(std::make_shared<AllocationPoolState>(backend, target_bytes, reuse)) {
    pools.erase(std::remove_if(pools.begin(), pools.end(), [](const auto& pool) { return pool.expired(); }), pools.end());
    pools.push_back(state_);
}
AllocationPool::~AllocationPool() {
    state_->clear(true);
    pools.erase(std::remove_if(pools.begin(), pools.end(), [&](const auto& pool) {
        auto value = pool.lock(); return !value || value == state_;
    }), pools.end());
}
AllocationPoolStats AllocationPool::stats() const { return state_->stats(); }
void AllocationPool::clear() { state_->clear(); }

size_t reclaim_cached_allocations(Backend backend) {
    size_t released = 0;
    const int device = detail::allocation_device(backend);
    for (auto& weak : pools) {
        auto pool = weak.lock();
        if (pool && pool->backend == backend && pool->device == device) released += pool->clear();
    }
    return released;
}
void prepare_cached_allocation(Backend backend, size_t bytes) {
    const int device = detail::allocation_device(backend);
    size_t target = std::numeric_limits<size_t>::max();
    bool limited = false;
    for (auto& weak : pools) {
        auto pool = weak.lock();
        if (pool && pool->backend == backend && pool->device == device && pool->target && pool->active()) {
            target = std::min(target, pool->target);
            limited = true;
        }
    }
    if (!limited) return;
    auto current = memory_stats(backend).live_bytes();
    if (current > target || bytes > target - current) reclaim_cached_allocations(backend);
}

BufferPtr detail::pooled_allocate(Backend backend, size_t count, DType dtype, MemoryKind kind,
                                  AllocateBuffer allocate, ClearBuffer clear) {
    auto pool = pools.empty() ? std::shared_ptr<AllocationPoolState>{} : current(backend, allocation_device(backend));
    auto allocation = pool ? pool->take(kind, dtype, count) : std::shared_ptr<Allocation>{};
    bool reused = bool(allocation);
    if (!allocation) {
        auto create = [&] { return std::make_shared<Allocation>(backend, count, dtype, kind, allocate); };
        try { allocation = create(); }
        catch (const std::bad_alloc&) {
            if (!reclaim_cached_allocations(backend)) throw;
            allocation = create();
        } catch (const AllocationFailure&) {
            if (!reclaim_cached_allocations(backend)) throw;
            allocation = create();
        }
        if (pool) pool->added(allocation);
    }
    if (!pool) return BufferPtr(allocation, allocation->buffer.get());
    std::shared_ptr<Lease> lease;
    try { lease = std::make_shared<Lease>(pool, allocation); }
    catch (...) { pool->release(std::move(allocation), true); throw; }
    if (reused && kind == MemoryKind::Tensor) {
        allocation->buffer->version = 0;
        try { if (clear) clear(allocation->buffer); }
        catch (...) { lease->recyclable = false; throw; }
    }
    return BufferPtr(lease, allocation->buffer.get());
}
} // namespace elman
