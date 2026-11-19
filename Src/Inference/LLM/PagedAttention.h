#pragma once
#include "Core/Tensor.h"

namespace elman {
class Qwen2PagedGraph;
// Immutable logical-to-physical slots. A missing slot represents literal zero
// padding, while a masked real slot retains its value for finite-mask semantics.
class PagedAttentionMap {
public:
    PagedAttentionMap(int batch, int physical_slots, const std::vector<int32_t>& slots,
                      const std::vector<int32_t>& visible, Backend backend, bool allow_contiguous = true);
    int batch() const { return batch_; }
    int length() const { return length_; }
private:
    int batch_, length_, physical_slots_;
    Tensor encoded_;
    bool contiguous_ = false;
    void update_graph_slots(const std::vector<int32_t>& slots);
    friend class Qwen2PagedGraph;
    friend Tensor paged_attention(const Tensor&, const Tensor&, const Tensor&, const PagedAttentionMap&,
                                  int, float, int, bool, int);
};

// Q is [batch,heads,queries,dimension]; physical K/V are
// [physical_slots,kv_heads*dimension]. This inference operator does not gather K/V.
Tensor paged_attention(const Tensor& query, const Tensor& keys, const Tensor& values,
                       const PagedAttentionMap& map, int kv_heads, float divisor,
                       int query_tile = 32, bool causal = true, int causal_offset = 0);
} // namespace elman
