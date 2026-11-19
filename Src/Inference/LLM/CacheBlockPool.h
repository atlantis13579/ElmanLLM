#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

namespace elman {
struct CacheBlock {
    int index = -1;
    uint64_t generation = 0;
    uint64_t pool = 0;
    bool operator==(const CacheBlock& other) const {
        return index == other.index && generation == other.generation && pool == other.pool;
    }
    bool operator!=(const CacheBlock& other) const { return !(*this == other); }
};

struct CacheBlockStats {
    size_t capacity = 0, live = 0, peak = 0;
    size_t references = 0, shared = 0;
};
class CacheCapacityExceeded : public std::runtime_error {
public:
    CacheCapacityExceeded(size_t requested, size_t available);
    size_t requested() const { return requested_; }
    size_t available() const { return available_; }
private:
    size_t requested_, available_;
};

struct CacheBlockState;
class CacheBlockPool;

// A move-only request lease. Growth reserves every requested block or changes
// nothing. Shrinking preserves the prefix and releases its tail references;
// only the last release invalidates a handle. Const operations can run together;
// mutations require external exclusion from access to the same lease/vector.
class CacheBlockLease {
public:
    CacheBlockLease() = default;
    ~CacheBlockLease();
    CacheBlockLease(const CacheBlockLease&) = delete;
    CacheBlockLease& operator=(const CacheBlockLease&) = delete;
    CacheBlockLease(CacheBlockLease&& other) noexcept;
    CacheBlockLease& operator=(CacheBlockLease&& other) noexcept;
    const std::vector<CacheBlock>& blocks() const { return blocks_; }
    size_t size() const { return blocks_.size(); }
    bool owns(const CacheBlock& block) const;
    // Share only committed, immutable payloads. Writing a shared block requires
    // a private replacement; these identities do not enforce payload access.
    CacheBlockLease share_prefix(size_t count) const;
    bool shared(size_t position) const;
    // Exchange ownership entries without changing total references. Both leases
    // must belong to one pool; duplicate entries within a lease are rejected.
    void swap_block(size_t position, CacheBlockLease& other, size_t other_position);
    void resize(size_t count);
    void clear() noexcept;
private:
    explicit CacheBlockLease(std::shared_ptr<CacheBlockState> state) : state_(std::move(state)) {}
    std::shared_ptr<CacheBlockState> state_;
    std::vector<CacheBlock> blocks_;
    friend class CacheBlockPool;
};

// Bounded physical block identities, independent of tensor payload allocation.
// Live leases retain their pool state. Handles include a pool identity and a
// reuse generation; an old or foreign handle never validates as a new block.
class CacheBlockPool {
public:
    explicit CacheBlockPool(int capacity);
    CacheBlockPool(const CacheBlockPool&) = delete;
    CacheBlockPool& operator=(const CacheBlockPool&) = delete;
    CacheBlockLease acquire(size_t count);
    bool live(const CacheBlock& block) const;
    CacheBlockStats stats() const;
private:
    std::shared_ptr<CacheBlockState> state_;
};
} // namespace elman
