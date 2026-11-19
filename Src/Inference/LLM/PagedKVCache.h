#pragma once

#include "Core/Tensor.h"
#include "Core/Memory.h"
#include "Inference/LLM/CacheBlockPool.h"
#include "Inference/LLM/PagedAttention.h"
#include <memory>

namespace elman {
struct PagedKVShape {
    int layers, heads, dimension, block_size, blocks;
};
struct PagedKVLayer { Tensor key, value; };
struct PagedDecodeBindings {
    std::vector<PagedKVLayer> layers;
    std::vector<int32_t> positions, destinations, slots;
    int physical_slots = 0;
};
struct PagedKVState;
class PagedKVAppend;
class PagedKVBatch;

// Fixed physical payload storage with reference-counted request leases. Host
// submissions using one pool are serialized; device dependencies survive reuse.
class PagedKVPool {
public:
    PagedKVPool(PagedKVShape shape, Backend backend, DType dtype);
    CacheBlockStats stats() const;
    PagedKVShape shape() const;
    Backend backend() const;
    DType dtype() const;
    static size_t payload_bytes(PagedKVShape shape, Backend backend, DType dtype);
    static size_t metadata_bytes(PagedKVShape shape);
private:
    std::shared_ptr<PagedKVState> state_;
    friend class PagedKVCache;
};

// A request owns all batch rows for one model. Reset cancels the request and
// drops its page references. The pool can outlive its public handle via leases.
// An active append must be destroyed before its cache; do not reenter the cache
// or its pool from an append callback (the transaction holds their mutex).
class PagedKVCache {
public:
    PagedKVCache(PagedKVPool& pool, int batch, int max_length);
    ~PagedKVCache();
    PagedKVCache(const PagedKVCache&) = delete;
    PagedKVCache& operator=(const PagedKVCache&) = delete;
    int length() const;
    int batch() const { return batch_; }
    int max_length() const { return max_length_; }
    PagedKVShape shape() const;
    Backend backend() const;
    DType dtype() const;
    void reset();
    bool belongs_to(const PagedKVPool& pool) const { return state_ == pool.state_; }
    size_t exclusive_blocks() const;
    // Share a committed prefix without copying payloads, including partial
    // pages. Subsequent appends isolate shared tails before writing. The fork
    // retains its pool and prefix after this cache is reset or destroyed.
    std::unique_ptr<PagedKVCache> fork(int prefix_length, int max_length) const;
    PagedKVLayer read(int layer) const;
    std::unique_ptr<PagedKVAppend> append(int tokens);
private:
    PagedKVCache(std::shared_ptr<PagedKVState> state, int batch, int max_length);
    std::shared_ptr<PagedKVState> state_;
    MemoryReservation metadata_;
    CacheBlockLease blocks_;
    int batch_, max_length_, length_ = 0;
    friend class PagedKVAppend;
    friend class PagedKVBatch;
};

// A borrowed view of distinct single-row requests from one physical pool.
// Requests must outlive the view and its append. Histories may have different
// lengths; a batched append left-pads shorter rows with zeros for attention.
// Every row consumes the same number of new tokens. Page/length publication is
// atomic across the whole batch; logical request caches never contain padding.
class PagedKVBatch {
public:
    explicit PagedKVBatch(std::vector<PagedKVCache*> requests);
    PagedKVBatch(const PagedKVBatch&) = delete;
    PagedKVBatch& operator=(const PagedKVBatch&) = delete;
    int batch() const { return int(requests_.size()); }
    int length() const;
    int max_length() const;
    std::vector<int> lengths() const;
    PagedKVShape shape() const;
    Backend backend() const;
    DType dtype() const;
    std::unique_ptr<PagedKVAppend> append(int tokens);
private:
    MemoryReservation metadata_;
    std::vector<PagedKVCache*> requests_;
    std::shared_ptr<PagedKVState> state_;
    friend class PagedKVAppend;
};

// Reserve the complete append and private tail replacements before writing any
// layer. Uncommitted writes touch only private/invisible slots; destruction
// rolls ownership back, including partial-page copies. Commit
// waits for this thread's current stream before publishing the new length.
class PagedKVAppend {
public:
    ~PagedKVAppend();
    PagedKVAppend(const PagedKVAppend&) = delete;
    PagedKVAppend& operator=(const PagedKVAppend&) = delete;
    int previous() const;
    int total() const;
    // Host 0/1 [batch,total] structural mask for a batched request append.
    // Empty for the original uniform-cache API; explicit user masks still apply.
    const std::vector<int32_t>& padding_mask() const;
    // Disabling materialization returns physical storage for direct attention.
    PagedKVLayer write(int layer, const Tensor& key, const Tensor& value, bool materialize = true);
    PagedKVLayer read_written(int layer) const;
    PagedAttentionMap attention_map(const std::vector<int32_t>& visible) const;
    void commit();
private:
    struct Impl;
    struct BatchImpl;
    explicit PagedKVAppend(PagedKVCache& cache, int tokens);
    explicit PagedKVAppend(PagedKVBatch& batch, int tokens);
    PagedDecodeBindings prepare_graph(int capacity);
    void complete_graph();
    std::unique_ptr<Impl> impl_;
    std::unique_ptr<BatchImpl> batch_impl_;
    friend class PagedKVCache;
    friend class PagedKVBatch;
    friend class Qwen2PagedGraph;
};
} // namespace elman
