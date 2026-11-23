#include "Inference/LLM/PrefixCache.h"
#include <algorithm>
#include <limits>

namespace elman {
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(std::string("Prefix cache: ") + message);
}
bool covers(const std::vector<int32_t>& stored, const std::vector<int32_t>& prompt, size_t length) {
    return stored.size() >= length && std::equal(prompt.begin(), prompt.begin() + length, stored.begin());
}
}
size_t PrefixCache::metadata_bytes(PagedKVShape shape, size_t max_blocks, size_t max_entries) {
    require(shape.block_size > 0 && shape.blocks > 0 && max_blocks > 0 && max_blocks <= size_t(shape.blocks) && max_entries > 0,
            "invalid retention bounds");
    // Retained keys and a candidate coexist until publication. Forked page
    // lease metadata is reserved by PagedKVCache itself.
    return checked_memory_add(checked_memory_mul(max_entries, 256),
        checked_memory_mul(checked_memory_mul(max_blocks, size_t(shape.block_size)), 2 * sizeof(int32_t)));
}
PrefixCache::PrefixCache(std::shared_ptr<PagedKVPool> pool, size_t max_blocks, size_t max_entries)
    : pool_(std::move(pool)), max_blocks_(max_blocks), max_entries_(max_entries) {
    require(bool(pool_), "null physical pool");
    metadata_ = MemoryReservation(Backend::CPU, metadata_bytes(pool_->shape(), max_blocks, max_entries));
    entries_.reserve(max_entries);
}
void PrefixCache::check_owner() const {
    require(owner_ == std::this_thread::get_id(), "cache belongs to another worker");
}
PrefixCacheStats PrefixCache::stats() const { check_owner(); return stats_; }
std::pair<size_t, size_t> PrefixCache::find(const std::vector<int32_t>& prompt) const {
    check_owner();
    require(!prompt.empty(), "empty prompt");
    size_t selected = entries_.size(), length = 0, page = size_t(pool_->shape().block_size);
    const size_t maximum = (prompt.size() - 1) / page * page;
    // Prefer the most recently used entry when equal-length prefixes match.
    for (size_t i = entries_.size(); i > 0; --i) {
        const std::vector<int32_t>& tokens = entries_[i - 1].tokens;
        size_t common = 0, limit = std::min(maximum, tokens.size());
        while (common < limit && tokens[common] == prompt[common]) ++common;
        common = common / page * page;
        if (common > length) { length = common; selected = i - 1; }
    }
    return {selected, length};
}
int PrefixCache::matched_tokens(const std::vector<int32_t>& prompt) const { return int(find(prompt).second); }
PrefixMatch PrefixCache::lookup(const std::vector<int32_t>& prompt, int max_length) {
    require(max_length > 0 && prompt.size() <= size_t(max_length), "invalid destination context");
    std::pair<size_t, size_t> match = find(prompt); size_t selected = match.first, length = match.second;
    if (!length) { ++stats_.misses; return {}; }
    // Allocation failure leaves the LRU order and hit counters unchanged.
    std::unique_ptr<PagedKVCache> branch = entries_[selected].cache->fork(int(length), max_length);
    uint64_t id = entries_[selected].id;
    std::rotate(entries_.begin() + selected, entries_.begin() + selected + 1, entries_.end());
    ++stats_.hits; stats_.reused_tokens += length;
    return {std::move(branch), int(length), id};
}
PrefixMatch PrefixCache::lookup_if_possible(const std::vector<int32_t>& prompt, int max_length) {
    try { return lookup(prompt, max_length); }
    catch (const MemoryLimitExceeded&) {}
    catch (const AllocationFailure&) {}
    catch (const std::bad_alloc&) {}
    ++stats_.allocation_failures; ++stats_.misses; return {};
}
PrefixPublication PrefixCache::publish_if_possible(const std::vector<int32_t>& prompt, const PagedKVCache& source) {
    try { return publish(prompt, source); }
    catch (const MemoryLimitExceeded&) {}
    catch (const AllocationFailure&) {}
    catch (const std::bad_alloc&) {}
    ++stats_.allocation_failures; return {};
}
size_t PrefixCache::publication_bytes() const {
    return checked_memory_mul(checked_memory_mul(max_blocks_, 2), sizeof(CacheBlock));
}
void PrefixCache::erase(size_t index) {
    stats_.retained_blocks -= entries_[index].tokens.size() / size_t(pool_->shape().block_size);
    entries_.erase(entries_.begin() + index);
    stats_.entries = entries_.size(); ++stats_.evictions;
}
bool PrefixCache::evict_oldest() {
    check_owner();
    if (entries_.empty()) return false;
    erase(0); return true;
}
bool PrefixCache::reclaim_one() {
    check_owner();
    for(size_t i=0;i<entries_.size();++i)if(entries_[i].cache->exclusive_blocks()) {erase(i);return true;}
    return false;
}
void PrefixCache::clear() { check_owner(); while (!entries_.empty()) erase(entries_.size() - 1); }
PrefixPublication PrefixCache::publish(const std::vector<int32_t>& prompt, const PagedKVCache& source) {
    check_owner();
    require(source.belongs_to(*pool_) && source.batch() == 1, "source belongs to another pool or has multiple rows");
    const size_t committed = size_t(source.length()), page = size_t(pool_->shape().block_size);
    require(committed <= prompt.size(), "source exceeds prompt history");
    const size_t blocks = std::min(committed / page, max_blocks_), length = blocks * page;
    if (!length) return {};
    for (size_t i = entries_.size(); i > 0; --i)
        if (covers(entries_[i - 1].tokens, prompt, length)) return {entries_[i - 1].id, int(length), false};
    require(next_id_ != std::numeric_limits<uint64_t>::max(), "entry identity exhausted");
    Entry candidate{next_id_, std::vector<int32_t>(prompt.begin(), prompt.begin() + length), source.fork(int(length), int(length))};
    // Build the complete candidate before dropping any existing entry. Moving
    // entries within the pre-reserved vector cannot allocate during publication.
    for (size_t i = entries_.size(); i > 0; --i)
        if (covers(candidate.tokens, entries_[i - 1].tokens, entries_[i - 1].tokens.size())) erase(i - 1);
    while (entries_.size() == max_entries_ || blocks > max_blocks_ - stats_.retained_blocks) erase(0);
    entries_.push_back(std::move(candidate));
    stats_.entries = entries_.size(); stats_.retained_blocks += blocks; ++stats_.publications;
    return {next_id_++, int(length), true};
}
} // namespace elman
