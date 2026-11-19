#pragma once

#include "Inference/LLM/PagedKVCache.h"
#include <thread>

namespace elman {
struct PrefixMatch {
    std::unique_ptr<PagedKVCache> cache;
    int tokens = 0;
    uint64_t entry = 0;
};
struct PrefixPublication {
    uint64_t entry = 0;
    int tokens = 0;
    bool inserted = false;
};
struct PrefixCacheStats {
    size_t entries = 0, retained_blocks = 0;
    uint64_t hits = 0, misses = 0, reused_tokens = 0, publications = 0, evictions = 0, allocation_failures = 0;
};

// Worker-owned exact token prefixes for one immutable model and physical pool.
// Only complete pages are retained. A lookup leaves at least one prompt token
// to execute, so the caller obtains fresh logits and independent sampling state.
// Entry eviction drops references; already returned request forks stay valid.
class PrefixCache {
public:
    PrefixCache(std::shared_ptr<PagedKVPool> pool, size_t max_blocks, size_t max_entries);
    PrefixCache(const PrefixCache&) = delete;
    PrefixCache& operator=(const PrefixCache&) = delete;
    PrefixMatch lookup(const std::vector<int32_t>& prompt, int max_length);
    PrefixPublication publish(const std::vector<int32_t>& prompt, const PagedKVCache& source);
    int matched_tokens(const std::vector<int32_t>& prompt) const;
    PrefixMatch lookup_if_possible(const std::vector<int32_t>& prompt, int max_length);
    PrefixPublication publish_if_possible(const std::vector<int32_t>& prompt, const PagedKVCache& source);
    size_t publication_bytes() const;
    bool evict_oldest();
    // Admission pressure must not discard a pinned prefix when doing so cannot
    // release any payload page. Metadata-capacity eviction still uses strict LRU.
    bool reclaim_one();
    void clear();
    PrefixCacheStats stats() const;
    bool belongs_to(const PagedKVPool& pool) const { return pool_.get() == &pool; }
    static size_t metadata_bytes(PagedKVShape shape, size_t max_blocks, size_t max_entries);
private:
    struct Entry {
        uint64_t id;
        std::vector<int32_t> tokens;
        std::unique_ptr<PagedKVCache> cache;
    };
    void check_owner() const;
    std::pair<size_t, size_t> find(const std::vector<int32_t>& prompt) const;
    void erase(size_t index);
    std::shared_ptr<PagedKVPool> pool_;
    MemoryReservation metadata_;
    size_t max_blocks_, max_entries_;
    std::vector<Entry> entries_;
    PrefixCacheStats stats_;
    uint64_t next_id_ = 1;
    std::thread::id owner_ = std::this_thread::get_id();
};
} // namespace elman
