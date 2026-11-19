#include "Inference/LLM/PagedKVCache.h"
#include "Core/Execution.h"
#include "Core/Graph.h"
#include <algorithm>
#include <limits>
#include <mutex>
#include <unordered_set>

namespace elman {
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(std::string("Paged KV cache: ") + message);
}
size_t product(size_t a, size_t b) { return checked_memory_mul(a, b); }
void validate(PagedKVShape s) {
    require(s.layers > 0 && s.heads > 0 && s.dimension > 0 && s.block_size > 0 && s.blocks >= 0,
            "invalid dimensions");
    auto width = product(size_t(s.heads), size_t(s.dimension));
    require(width <= size_t(std::numeric_limits<int>::max()), "head width exceeds Int32");
    require(product(product(size_t(s.blocks), size_t(s.block_size)), width) <= size_t(std::numeric_limits<int>::max()),
            "physical layer exceeds Int32");
}
size_t block_count(int length, int block_size, int batch) {
    return product((size_t(length) + size_t(block_size) - 1) / size_t(block_size), size_t(batch));
}
}

struct PagedKVState {
    PagedKVShape shape;
    Backend backend;
    DType dtype;
    MemoryReservation metadata;
    CacheBlockPool pool;
    std::mutex mutex;
    std::vector<PagedKVLayer> layers;
    PagedKVState(PagedKVShape s, Backend b, DType d)
        : shape(s), backend(b), dtype(d), metadata(Backend::CPU, PagedKVPool::metadata_bytes(s)), pool(s.blocks) {
        layers.reserve(size_t(s.layers));
        graph::RecordingGuard guard;
        for (int i = 0; i < s.layers; ++i)
            layers.push_back({Tensor::zeros({s.blocks * s.block_size, s.heads * s.dimension}, b, false, d),
                              Tensor::zeros({s.blocks * s.block_size, s.heads * s.dimension}, b, false, d)});
    }
    Tensor mapping(const CacheBlockLease& blocks, int batch, int start, int end) const {
        const int count = end - start;
        std::vector<int32_t> ids(product(size_t(batch), size_t(count)));
        for (int b = 0; b < batch; ++b) for (int t = start; t < end; ++t)
            ids[size_t(b) * count + t - start] = blocks.blocks()[size_t(t / shape.block_size) * batch + b].index *
                shape.block_size + t % shape.block_size;
        return Tensor::from_int_data({int(ids.size())}, ids, backend);
    }
    PagedKVLayer gather(int layer, int batch, int length, const Tensor& ids) const {
        auto read = [&](const Tensor& data) {
            return transpose(reshape(index_select(data, 0, ids), {batch, length, shape.heads, shape.dimension}), 1, 2);
        };
        return {read(layers[size_t(layer)].key), read(layers[size_t(layer)].value)};
    }
};

size_t PagedKVPool::metadata_bytes(PagedKVShape s) {
    validate(s);
    // Includes block ownership, reuse generations, the free list and payload
    // descriptors. Request leases and slot mappings are reserved separately.
    return checked_memory_add(product(size_t(s.blocks), 64), product(size_t(s.layers), sizeof(PagedKVLayer)));
}
size_t PagedKVPool::payload_bytes(PagedKVShape s, Backend backend, DType dtype) {
    validate(s);
    require(is_compute_float(dtype), "unsupported payload dtype");
    auto elements = product(product(size_t(s.blocks), size_t(s.block_size)), product(size_t(s.heads), size_t(s.dimension)));
    return product(product(2, size_t(s.layers)), memory_allocation_bytes(backend, dtype_storage_bytes(dtype, elements)));
}
PagedKVPool::PagedKVPool(PagedKVShape s, Backend backend, DType dtype) {
    payload_bytes(s, backend, dtype);
    state_ = std::make_shared<PagedKVState>(s, backend, dtype);
}
CacheBlockStats PagedKVPool::stats() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    return state_->pool.stats();
}
PagedKVShape PagedKVPool::shape() const { return state_->shape; }
Backend PagedKVPool::backend() const { return state_->backend; }
DType PagedKVPool::dtype() const { return state_->dtype; }
PagedKVCache::PagedKVCache(PagedKVPool& pool, int batch, int max_length)
    : PagedKVCache(pool.state_, batch, max_length) {}
PagedKVCache::PagedKVCache(std::shared_ptr<PagedKVState> state, int batch, int max_length)
    : state_(std::move(state)), batch_(batch), max_length_(max_length) {
    require(batch > 0 && max_length > 0, "invalid request dimensions");
    auto s = state_->shape;
    require(product(product(size_t(batch), size_t(max_length)), product(size_t(s.heads), size_t(s.dimension))) <=
            size_t(std::numeric_limits<int>::max()), "logical layer exceeds Int32");
    metadata_ = MemoryReservation(Backend::CPU, product(std::min(size_t(s.blocks), block_count(max_length, s.block_size, batch)), sizeof(CacheBlock)));
    blocks_ = state_->pool.acquire(0);
}
PagedKVCache::~PagedKVCache() { reset(); }
int PagedKVCache::length() const { std::lock_guard<std::mutex> lock(state_->mutex); return length_; }
size_t PagedKVCache::exclusive_blocks() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    size_t count=0;for(size_t i=0;i<blocks_.size();++i)if(!blocks_.shared(i))++count;
    return count;
}
PagedKVShape PagedKVCache::shape() const { return state_->shape; }
Backend PagedKVCache::backend() const { return state_->backend; }
DType PagedKVCache::dtype() const { return state_->dtype; }
void PagedKVCache::reset() {
    std::lock_guard<std::mutex> lock(state_->mutex);
    blocks_.clear();
    length_ = 0;
}
std::unique_ptr<PagedKVCache> PagedKVCache::fork(int prefix_length, int max_length) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(prefix_length >= 0 && prefix_length <= length_ && prefix_length <= max_length,
            "shared prefix exceeds committed length or destination context");
    auto shared = blocks_.share_prefix(block_count(prefix_length, state_->shape.block_size, batch_));
    // No potentially throwing operation may follow construction while holding
    // the pool lock: the cache destructor acquires that same lock.
    auto result = std::unique_ptr<PagedKVCache>(new PagedKVCache(state_, batch_, max_length));
    result->blocks_ = std::move(shared);
    result->length_ = prefix_length;
    return result;
}
PagedKVLayer PagedKVCache::read(int layer) const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    require(layer >= 0 && layer < state_->shape.layers, "layer index outside pool");
    graph::RecordingGuard guard;
    MemoryReservation mapping(Backend::CPU, product(product(size_t(batch_), size_t(length_)), 8));
    return state_->gather(layer, batch_, length_, state_->mapping(blocks_, batch_, 0, length_));
}

PagedKVBatch::PagedKVBatch(std::vector<PagedKVCache*> requests)
    : metadata_(Backend::CPU, product(requests.size(), 64)), requests_(std::move(requests)) {
    require(!requests_.empty() && requests_.size() <= size_t(std::numeric_limits<int>::max()), "empty or oversized request batch");
    require(requests_.front() != nullptr, "null request in batch");
    state_ = requests_.front()->state_;
    std::unordered_set<PagedKVCache*> seen;
    for (auto* request : requests_) {
        require(request != nullptr && request->state_ == state_ && request->batch_ == 1,
                "batched requests must be single rows from one pool");
        require(seen.insert(request).second, "duplicate request in batch");
    }
}
int PagedKVBatch::length() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    int result = 0;
    for (auto* request : requests_) result = std::max(result, request->length_);
    return result;
}
int PagedKVBatch::max_length() const {
    int result = 0;
    for (auto* request : requests_) result = std::max(result, request->max_length_);
    return result;
}
std::vector<int> PagedKVBatch::lengths() const {
    std::lock_guard<std::mutex> lock(state_->mutex);
    std::vector<int> result;
    for (auto* request : requests_) result.push_back(request->length_);
    return result;
}
PagedKVShape PagedKVBatch::shape() const { return state_->shape; }
Backend PagedKVBatch::backend() const { return state_->backend; }
DType PagedKVBatch::dtype() const { return state_->dtype; }

namespace {
struct TailOwner { CacheBlockLease* blocks; int batch, length; };
bool shared_tail(const TailOwner& owner, int page) {
    if (owner.length % page == 0) return false;
    size_t start = size_t(owner.length / page) * owner.batch;
    for (int row = 0; row < owner.batch; ++row)
        if (owner.blocks->shared(start + row)) return true;
    return false;
}
// The pool mutex excludes forks and other appends during the transaction.
// Old shared handles remain leased until commit, so rollback needs no allocation.
struct PagedKVTails {
    struct Entry { CacheBlockLease* blocks; size_t position; int tokens; };
    std::shared_ptr<PagedKVState> state;
    MemoryReservation metadata;
    std::vector<Entry> entries;
    std::vector<CacheBlockLease> retired;
    Tensor read_ids, write_ids;
    size_t exchanged = 0;
    bool committed = false;
    PagedKVTails(std::shared_ptr<PagedKVState> state_, const std::vector<TailOwner>& owners)
        : state(std::move(state_)) {
        size_t rows = 0;
        for (const auto& owner : owners) rows = checked_memory_add(rows, size_t(owner.batch));
        const int page = state->shape.block_size;
        metadata = MemoryReservation(Backend::CPU, checked_memory_add(product(rows, 128),
            product(product(rows, size_t(page - 1)), 32)));
        entries.reserve(rows);
        for (const auto& owner : owners) {
            if (owner.length % page == 0) continue;
            size_t start = size_t(owner.length / page) * owner.batch;
            for (int row = 0; row < owner.batch; ++row)
                if (owner.blocks->shared(start + row)) entries.push_back({owner.blocks, start + row, owner.length % page});
        }
        auto stats = state->pool.stats();
        if (entries.size() > stats.capacity - stats.live)
            throw CacheCapacityExceeded(entries.size(), stats.capacity - stats.live);
        retired.reserve(entries.size());
        // Two rows can retain the same old page. Separate leases keep each
        // logical ownership vector unique while preserving both references.
        for (size_t i = 0; i < entries.size(); ++i) retired.push_back(state->pool.acquire(1));
        std::vector<int32_t> reads, writes;
        reads.reserve(product(entries.size(), size_t(page - 1))); writes.reserve(reads.capacity());
        for (size_t i = 0; i < entries.size(); ++i) {
            auto& entry = entries[i];
            int from = entry.blocks->blocks()[entry.position].index * page;
            int to = retired[i].blocks()[0].index * page;
            for (int token = 0; token < entry.tokens; ++token) { reads.push_back(from + token); writes.push_back(to + token); }
        }
        read_ids = Tensor::from_int_data({int(reads.size())}, reads, state->backend);
        write_ids = Tensor::from_int_data({int(writes.size())}, writes, state->backend);
        try {
            for (size_t i = 0; i < entries.size(); ++i) {
                entries[i].blocks->swap_block(entries[i].position, retired[i], 0); ++exchanged;
            }
        } catch (...) { rollback(); throw; }
    }
    void rollback() {
        while (exchanged) {
            size_t i = --exchanged;
            entries[i].blocks->swap_block(entries[i].position, retired[i], 0);
        }
    }
    ~PagedKVTails() { if (!committed) rollback(); }
    void copy(int layer) {
        auto& payload = state->layers[size_t(layer)];
        for (auto* data : {&payload.key, &payload.value})
            data->assign_indexed(0, write_ids, index_select(*data, 0, read_ids));
    }
};
}

struct PagedKVAppend::BatchImpl {
    std::shared_ptr<PagedKVState> state;
    std::unique_lock<std::mutex> lock;
    MemoryReservation metadata;
    std::vector<PagedKVCache*> requests;
    std::vector<size_t> original_blocks;
    std::vector<int> lengths;
    std::vector<uint8_t> written;
    std::vector<int32_t> mask;
    Tensor write_ids, read_ids, valid, zero;
    std::unique_ptr<PagedKVTails> tails;
    std::shared_ptr<detail::StreamState> stream;
    int batch, previous = 0, total;
    bool committed = false;

    BatchImpl(PagedKVBatch& view, int tokens) : state(view.state_), lock(state->mutex), batch(view.batch()) {
        require(tokens > 0, "append token count must be positive");
        for (auto* request : view.requests_) {
            require(tokens <= request->max_length_ - request->length_, "batch append exceeds a request context");
            previous = std::max(previous, request->length_);
        }
        total = previous + tokens;
        auto& s = *state;
        auto rows = product(size_t(batch), size_t(total));
        require(product(rows, product(size_t(s.shape.heads), size_t(s.shape.dimension))) <= size_t(std::numeric_limits<int>::max()),
                "batched logical layer exceeds Int32");
        metadata = MemoryReservation(Backend::CPU, checked_memory_add(checked_memory_add(product(size_t(batch), 64),
            product(product(size_t(batch), size_t(tokens) + size_t(total)), 16)), size_t(s.shape.layers)));
        requests = view.requests_;
        original_blocks.reserve(requests.size());
        lengths.reserve(requests.size());
        for (auto* request : requests) {
            original_blocks.push_back(request->blocks_.size());
            lengths.push_back(request->length_);
        }
        written.resize(size_t(s.shape.layers));
        stream = detail::stream_binding(s.backend).stream;
        try {
            for (auto* request : requests)
                request->blocks_.resize(block_count(request->length_ + tokens, s.shape.block_size, 1));
            bool shared = false;
            for (auto* request : requests) shared |= shared_tail({&request->blocks_, 1, request->length_}, s.shape.block_size);
            if (shared) {
                std::vector<TailOwner> owners;
                for (auto* request : requests) owners.push_back({&request->blocks_, 1, request->length_});
                tails = std::make_unique<PagedKVTails>(state, owners);
            }
            std::vector<int32_t> reads(rows, 0), writes(product(size_t(batch), size_t(tokens)));
            mask.resize(rows, 0);
            bool padding = false;
            for (int b = 0; b < batch; ++b) {
                auto& blocks = requests[size_t(b)]->blocks_.blocks();
                int offset = previous - lengths[size_t(b)];
                padding |= offset != 0;
                for (int t = 0; t < lengths[size_t(b)] + tokens; ++t) {
                    int slot = blocks[size_t(t / s.shape.block_size)].index * s.shape.block_size + t % s.shape.block_size;
                    size_t index = size_t(b) * total + offset + t;
                    reads[index] = slot;
                    mask[index] = 1;
                    if (t >= lengths[size_t(b)]) writes[size_t(b) * tokens + t - lengths[size_t(b)]] = slot;
                }
            }
            write_ids = Tensor::from_int_data({int(writes.size())}, writes, s.backend);
            read_ids = Tensor::from_int_data({int(reads.size())}, reads, s.backend);
            if (padding) {
                valid = Tensor::from_int_data({batch, 1, total, 1}, mask, s.backend, DType::Bool);
                zero = Tensor::zeros({}, s.backend, false, s.dtype);
            }
        } catch (...) { rollback(); throw; }
    }
    void rollback() {
        tails.reset();
        for (size_t i = 0; i < requests.size(); ++i) requests[i]->blocks_.resize(original_blocks[i]);
    }
    ~BatchImpl() { if (!committed) rollback(); }
    void active() const {
        require(!committed, "transaction already committed");
        require(detail::stream_binding(state->backend).stream == stream, "transaction changed execution stream");
    }
    PagedKVLayer gather(int layer) const {
        auto result = state->gather(layer, batch, total, read_ids);
        if (valid.defined()) {
            result.key = where(valid, result.key, zero);
            result.value = where(valid, result.value, zero);
        }
        return result;
    }
    PagedKVLayer write(int layer, const Tensor& key, const Tensor& value, bool materialize) {
        active();
        auto& s = *state;
        require(layer >= 0 && layer < s.shape.layers && !written[size_t(layer)], "invalid or repeated layer write");
        int tokens = total - previous;
        for (auto tensor : {&key, &value})
            require(tensor->defined() && tensor->shape() == Shape{batch, s.shape.heads, tokens, s.shape.dimension} &&
                    tensor->backend() == s.backend && tensor->dtype() == s.dtype && !tensor->requires_grad(),
                    "layer shape, backend, dtype or gradient differs");
        graph::RecordingGuard guard;
        if (tails) tails->copy(layer);
        auto scatter = [&](Tensor& destination, const Tensor& source) {
            destination.assign_indexed(0, write_ids, reshape(transpose(source, 1, 2),
                {batch * tokens, s.shape.heads * s.shape.dimension}));
        };
        scatter(s.layers[size_t(layer)].key, key);
        scatter(s.layers[size_t(layer)].value, value);
        // Padding must not disclose whichever request owns physical slot zero.
        // Select literal zero instead of multiplying possibly nonfinite payloads.
        auto result = materialize ? gather(layer) : s.layers[size_t(layer)];
        written[size_t(layer)] = 1;
        return result;
    }
    void commit() {
        active();
        require(std::all_of(written.begin(), written.end(), [](uint8_t value) { return value != 0; }), "append has unwritten layers");
        if (stream) stream->record()->synchronize();
        for (size_t i = 0; i < requests.size(); ++i) requests[i]->length_ = lengths[i] + total - previous;
        if (tails) tails->committed = true;
        committed = true;
    }
};

struct PagedKVAppend::Impl {
    PagedKVCache& cache;
    std::unique_lock<std::mutex> lock;
    size_t original_blocks;
    int previous, total;
    MemoryReservation metadata;
    std::vector<uint8_t> written;
    Tensor write_ids, read_ids;
    std::unique_ptr<PagedKVTails> tails;
    std::shared_ptr<detail::StreamState> stream;
    bool committed = false;
    Impl(PagedKVCache& c, int tokens) : cache(c), lock(c.state_->mutex), original_blocks(c.blocks_.size()),
        previous(c.length_), total(previous) {
        require(tokens > 0 && tokens <= c.max_length_ - previous, "append exceeds request context");
        total += tokens;
        auto& s = *c.state_;
        metadata = MemoryReservation(Backend::CPU, checked_memory_add(product(product(size_t(c.batch_), size_t(tokens) + total), 8), size_t(s.shape.layers)));
        written.resize(size_t(s.shape.layers));
        stream = detail::stream_binding(s.backend).stream;
        // A failed constructor must restore any pages reserved before a tensor
        // allocation or upload failed. The destructor cannot run in that case.
        try {
            c.blocks_.resize(block_count(total, s.shape.block_size, c.batch_));
            if (shared_tail({&c.blocks_, c.batch_, previous}, s.shape.block_size))
                tails = std::make_unique<PagedKVTails>(c.state_, std::vector<TailOwner>{{&c.blocks_, c.batch_, previous}});
            write_ids = s.mapping(c.blocks_, c.batch_, previous, total);
            read_ids = s.mapping(c.blocks_, c.batch_, 0, total);
        } catch (...) { tails.reset(); c.blocks_.resize(original_blocks); throw; }
    }
    ~Impl() { if (!committed) { tails.reset(); cache.blocks_.resize(original_blocks); } }
    void active() const {
        require(!committed, "transaction already committed");
        require(detail::stream_binding(cache.backend()).stream == stream, "transaction changed execution stream");
    }
};
PagedKVAppend::PagedKVAppend(PagedKVCache& cache, int tokens) : impl_(std::make_unique<Impl>(cache, tokens)) {}
PagedKVAppend::PagedKVAppend(PagedKVBatch& batch, int tokens) : batch_impl_(std::make_unique<BatchImpl>(batch, tokens)) {}
PagedKVAppend::~PagedKVAppend() = default;
std::unique_ptr<PagedKVAppend> PagedKVCache::append(int tokens) {
    return std::unique_ptr<PagedKVAppend>(new PagedKVAppend(*this, tokens));
}
std::unique_ptr<PagedKVAppend> PagedKVBatch::append(int tokens) {
    return std::unique_ptr<PagedKVAppend>(new PagedKVAppend(*this, tokens));
}
int PagedKVAppend::previous() const { return batch_impl_ ? batch_impl_->previous : impl_->previous; }
int PagedKVAppend::total() const { return batch_impl_ ? batch_impl_->total : impl_->total; }
const std::vector<int32_t>& PagedKVAppend::padding_mask() const {
    static const std::vector<int32_t> empty;
    return batch_impl_ ? batch_impl_->mask : empty;
}
PagedKVLayer PagedKVAppend::write(int layer, const Tensor& key, const Tensor& value, bool materialize) {
    if (batch_impl_) return batch_impl_->write(layer, key, value, materialize);
    auto& i = *impl_;
    i.active();
    auto& c = i.cache;
    auto& s = *c.state_;
    require(layer >= 0 && layer < s.shape.layers && !i.written[size_t(layer)], "invalid or repeated layer write");
    int tokens = i.total - i.previous;
    for (auto tensor : {&key, &value})
        require(tensor->defined() && tensor->shape() == Shape{c.batch_, s.shape.heads, tokens, s.shape.dimension} &&
                tensor->backend() == s.backend && tensor->dtype() == s.dtype && !tensor->requires_grad(),
                "layer shape, backend, dtype or gradient differs");
    graph::RecordingGuard guard;
    if (i.tails) i.tails->copy(layer);
    auto write = [&](Tensor& destination, const Tensor& source) {
        destination.assign_indexed(0, i.write_ids, reshape(transpose(source, 1, 2),
            {c.batch_ * tokens, s.shape.heads * s.shape.dimension}));
    };
    write(s.layers[size_t(layer)].key, key);
    write(s.layers[size_t(layer)].value, value);
    auto result = materialize ? s.gather(layer, c.batch_, i.total, i.read_ids) : s.layers[size_t(layer)];
    i.written[size_t(layer)] = 1;
    return result;
}
PagedKVLayer PagedKVAppend::read_written(int layer) const {
    if (batch_impl_) {
        auto& i = *batch_impl_;
        i.active();
        require(layer >= 0 && layer < i.state->shape.layers && i.written[size_t(layer)], "layer has not been written");
        return i.gather(layer);
    }
    auto& i = *impl_;
    i.active();
    auto& c = i.cache;
    require(layer >= 0 && layer < c.state_->shape.layers && i.written[size_t(layer)], "layer has not been written");
    return c.state_->gather(layer, c.batch_, i.total, i.read_ids);
}
PagedAttentionMap PagedKVAppend::attention_map(const std::vector<int32_t>& visible) const {
    const auto& ids = batch_impl_ ? batch_impl_->read_ids : impl_->read_ids;
    const auto& s = batch_impl_ ? *batch_impl_->state : *impl_->cache.state_;
    if (batch_impl_) batch_impl_->active(); else impl_->active();
    MemoryReservation temporary(Backend::CPU, product(ids.numel(), sizeof(int32_t)));
    auto slots = ids.int_data();
    if (batch_impl_) for (size_t i = 0; i < slots.size(); ++i)
        if (!batch_impl_->mask[i]) slots[i] = -1;
    return PagedAttentionMap(batch_impl_ ? batch_impl_->batch : impl_->cache.batch_,
                             s.shape.blocks * s.shape.block_size, slots, visible, s.backend);
}
PagedDecodeBindings PagedKVAppend::prepare_graph(int capacity) {
    require(bool(batch_impl_), "graph decode requires a paged batch append");
    auto& batch = *batch_impl_;
    batch.active();
    require(batch.total == batch.previous + 1 && capacity >= batch.total &&
            std::none_of(batch.written.begin(), batch.written.end(), [](uint8_t value) { return value != 0; }),
            "invalid decode graph append geometry");
    const auto& shape = batch.state->shape;
    PagedDecodeBindings result;
    result.positions.assign(batch.lengths.begin(), batch.lengths.end());
    result.destinations.resize(batch.batch);
    result.slots.assign(product(size_t(batch.batch), size_t(capacity)), -1);
    result.physical_slots = shape.blocks * shape.block_size;
    result.layers = batch.state->layers;
    for (int row = 0; row < batch.batch; ++row) {
        const auto& blocks = batch.requests[row]->blocks_.blocks();
        const int length = batch.lengths[row] + 1;
        for (int token = 0; token < length; ++token) {
            const int slot = blocks[token / shape.block_size].index * shape.block_size + token % shape.block_size;
            result.slots[size_t(row) * capacity + capacity - length + token] = slot;
            if (token == length - 1) result.destinations[row] = slot;
        }
    }
    for (int layer = 0; batch.tails && layer < shape.layers; ++layer) batch.tails->copy(layer);
    return result;
}
void PagedKVAppend::complete_graph() {
    require(bool(batch_impl_), "graph decode requires a paged batch append");
    batch_impl_->active();
    std::fill(batch_impl_->written.begin(), batch_impl_->written.end(), uint8_t{1});
    commit();
}
void PagedKVAppend::commit() {
    if (batch_impl_) { batch_impl_->commit(); return; }
    auto& i = *impl_;
    i.active();
    require(std::all_of(i.written.begin(), i.written.end(), [](uint8_t v) { return v != 0; }), "append has unwritten layers");
    if (i.stream) i.stream->record()->synchronize();
    i.cache.length_ = i.total;
    if (i.tails) i.tails->committed = true;
    i.committed = true;
}
} // namespace elman
