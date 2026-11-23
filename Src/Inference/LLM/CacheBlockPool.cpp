#include "Inference/LLM/CacheBlockPool.h"
#include <algorithm>
#include <atomic>
#include <limits>
#include <mutex>
#include <string>
#include <utility>

namespace elman {
namespace {
uint64_t pool_identity() {
    static std::atomic<uint64_t> next{1};
    uint64_t value = next.load(std::memory_order_relaxed);
    for (;;) {
        if (value == std::numeric_limits<uint64_t>::max())
            throw std::overflow_error("Cache block pool identities are exhausted");
        if (next.compare_exchange_weak(value, value + 1, std::memory_order_relaxed)) return value;
    }
}
}

struct CacheBlockState {
    struct Slot { uint64_t generation = 0; size_t references = 0; };
    const uint64_t identity = pool_identity();
    std::mutex mutex;
    std::vector<Slot> slots;
    std::vector<int> free;
    size_t peak = 0, references = 0, shared = 0;
    explicit CacheBlockState(int capacity) : slots(size_t(capacity)) {
        free.reserve(size_t(capacity));
        for (int index = capacity; index-- > 0;) free.push_back(index);
    }
    void release(int index) noexcept {
        CacheBlockState::Slot& slot = slots[size_t(index)];
        if (slot.references == 2) --shared;
        if (--slot.references == 0) free.push_back(index);
        --references;
    }
};

CacheCapacityExceeded::CacheCapacityExceeded(size_t requested, size_t available)
    : std::runtime_error("Cache blocks exhausted: requested=" + std::to_string(requested) +
                         ", available=" + std::to_string(available)),
      requested_(requested), available_(available) {}

CacheBlockPool::CacheBlockPool(int capacity) {
    if (capacity < 0) throw std::invalid_argument("Cache block capacity must be nonnegative");
    state_ = std::make_shared<CacheBlockState>(capacity);
}
CacheBlockLease CacheBlockPool::acquire(size_t count) {
    CacheBlockLease result(state_);
    result.resize(count);
    return result;
}
bool CacheBlockPool::live(const CacheBlock& block) const {
    if (block.pool != state_->identity || block.index < 0 || size_t(block.index) >= state_->slots.size()) return false;
    std::lock_guard<std::mutex> lock(state_->mutex);
    const CacheBlockState::Slot& slot = state_->slots[size_t(block.index)];
    return slot.references != 0 && slot.generation == block.generation;
}
CacheBlockStats CacheBlockPool::stats() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return {state_->slots.size(), state_->slots.size() - state_->free.size(), state_->peak, state_->references, state_->shared};
}

CacheBlockLease::~CacheBlockLease() { clear(); }
CacheBlockLease::CacheBlockLease(CacheBlockLease&& other) noexcept
    : state_(std::move(other.state_)), blocks_(std::move(other.blocks_)) {}
CacheBlockLease& CacheBlockLease::operator=(CacheBlockLease&& other) noexcept {
    if (this != &other) {
        clear();
        state_ = std::move(other.state_);
        blocks_ = std::move(other.blocks_);
    }
    return *this;
}
bool CacheBlockLease::owns(const CacheBlock& block) const {
    return std::find(blocks_.begin(), blocks_.end(), block) != blocks_.end();
}
CacheBlockLease CacheBlockLease::share_prefix(size_t count) const {
    if (!state_) throw std::logic_error("Cache block lease has no pool");
    if (count > blocks_.size()) throw std::invalid_argument("Shared block prefix exceeds the lease");
    CacheBlockLease result(state_);
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (count > std::numeric_limits<size_t>::max() - state_->references)
        throw std::overflow_error("Cache block reference count is exhausted");
    for (size_t i = 0; i < count; ++i)
        if (state_->slots[size_t(blocks_[i].index)].references == std::numeric_limits<size_t>::max())
            throw std::overflow_error("Cache block slot references are exhausted");
    // Copy host metadata before incrementing any physical references.
    result.blocks_.assign(blocks_.begin(), blocks_.begin() + count);
    for (const CacheBlock& block : result.blocks_) {
        CacheBlockState::Slot& slot = state_->slots[size_t(block.index)];
        if (slot.references++ == 1) ++state_->shared;
    }
    state_->references += count;
    return result;
}
bool CacheBlockLease::shared(size_t position) const {
    if (!state_) throw std::logic_error("Cache block lease has no pool");
    if (position >= blocks_.size()) throw std::out_of_range("Cache block position is outside the lease");
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->slots[size_t(blocks_[position].index)].references > 1;
}
void CacheBlockLease::swap_block(size_t position, CacheBlockLease& other, size_t other_position) {
    if (!state_ || state_ != other.state_) throw std::invalid_argument("Cache block exchange requires one pool");
    if (position >= blocks_.size() || other_position >= other.blocks_.size())
        throw std::out_of_range("Cache block exchange position is outside the lease");
    const CacheBlock a = blocks_[position], b = other.blocks_[other_position];
    if (this != &other && a != b && (owns(b) || other.owns(a)))
        throw std::invalid_argument("Cache block exchange would duplicate a lease entry");
    std::swap(blocks_[position], other.blocks_[other_position]);
}
void CacheBlockLease::resize(size_t count) {
    if (!state_) throw std::logic_error("Cache block lease has no pool");
    if (count == blocks_.size()) return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    if (count > blocks_.size()) {
        size_t extra = count - blocks_.size();
        if (extra > state_->free.size()) throw CacheCapacityExceeded(extra, state_->free.size());
        if (extra > std::numeric_limits<size_t>::max() - state_->references)
            throw std::overflow_error("Cache block reference count is exhausted");
        // Validate counters and allocate host metadata before changing ownership.
        for (size_t i = 0; i < extra; ++i)
            if (state_->slots[size_t(state_->free[state_->free.size() - i - 1])].generation ==
                std::numeric_limits<uint64_t>::max())
                throw std::overflow_error("Cache block reuse generation is exhausted");
        blocks_.reserve(count);
        while (blocks_.size() < count) {
            int index = state_->free.back();
            CacheBlockState::Slot& slot = state_->slots[size_t(index)];
            blocks_.push_back({index, slot.generation + 1, state_->identity});
            ++slot.generation;
            slot.references = 1;
            ++state_->references;
            state_->free.pop_back();
        }
        state_->peak = std::max(state_->peak, state_->slots.size() - state_->free.size());
    } else {
        while (blocks_.size() > count) {
            int index = blocks_.back().index;
            state_->release(index);
            blocks_.pop_back();
        }
        if (!count) std::vector<CacheBlock>().swap(blocks_);
    }
}
void CacheBlockLease::clear() noexcept {
    if (!state_ || blocks_.empty()) return;
    std::lock_guard<std::mutex> lock(state_->mutex);
    for (auto it = blocks_.rbegin(); it != blocks_.rend(); ++it) {
        state_->release(it->index);
    }
    std::vector<CacheBlock>().swap(blocks_);
}
} // namespace elman
