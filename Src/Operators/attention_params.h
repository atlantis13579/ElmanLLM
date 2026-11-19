#pragma once
#include "Backends/AttentionGemm.h"
#include "Operators/attributes.h"
#include <algorithm>

namespace elman {
struct TiledAttentionShape {
    DType dtype;
    int batch, heads, queries, keys, dimension, values, groups, tile;
    explicit TiledAttentionShape(const KernelParams& p)
        : dtype(DType(p.output_dtype())), batch(p.input_shape(KernelInput::A, 0)),
          heads(p.input_shape(KernelInput::A, 1)), queries(p.input_shape(KernelInput::A, 2)),
          keys(p.input_shape(KernelInput::B, 2)), dimension(p.input_shape(KernelInput::A, 3)),
          values(p.input_shape(KernelInput::C, 3)), groups(heads / p.input_shape(KernelInput::B, 1)),
          tile(std::min(queries, p.attribute(operator_attr::tiled_attention::kQueryTile))) {}
    AttentionGemm scores(int rows) const {
        return {dtype, rows, keys, dimension, batch * heads, groups,
                queries * dimension, keys * dimension, rows * keys, true};
    }
    AttentionGemm attended(int rows) const {
        return {dtype, rows, values, keys, batch * heads, groups,
                rows * keys, keys * values, queries * values, false};
    }
    KernelParams probabilities(const KernelParams& source, int rows, int start, bool one_head = false) const {
        KernelParams p;
        int shape[4] = {one_head ? 1 : batch, one_head ? 1 : heads, rows, keys};
        p.v[KernelParams::kOutputCountSlot] = shape[0] * shape[1] * rows * keys;
        p.v[KernelParams::kOutputRankSlot] = 4;
        for (auto input : {KernelInput::A, KernelInput::B, KernelInput::C, KernelInput::D})
            p.v[KernelParams::input_slot(input)] = -1;
        int base = KernelParams::input_slot(KernelInput::A), stride = 1;
        p.v[base] = 4;
        for (int d = 3; d >= 0; --d) {
            p.v[KernelParams::kOutputShapeSlot + d] = shape[d];
            p.v[base + KernelParams::kInputShapeOffset + d] = shape[d];
            p.v[base + KernelParams::kInputStrideOffset + d] = stride;
            stride *= shape[d];
        }
        base = KernelParams::input_slot(KernelInput::B);
        p.v[base] = source.input_rank(KernelInput::D);
        for (int d = 0; d < p.v[base]; ++d) {
            p.v[base + KernelParams::kInputShapeOffset + d] = source.input_shape(KernelInput::D, d);
            p.v[base + KernelParams::kInputStrideOffset + d] = source.input_stride(KernelInput::D, d);
        }
        p.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(dtype);
        p.v[KernelParams::input_dtype_slot(KernelInput::B)] = int(dtype);
        p.v[KernelParams::kOutputDTypeSlot] = int(dtype);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kAxis] = 3;
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kDivisor] = source.attribute(operator_attr::tiled_attention::kDivisor);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kCausal] = source.attribute(operator_attr::tiled_attention::kCausal);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kCausalOffset] = source.attribute(operator_attr::tiled_attention::kCausalOffset);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kQueryBase] = start;
        return p;
    }
    size_t mask_offset(const KernelParams& p, int start) const {
        int query_axis = p.input_rank(KernelInput::D) - 2;
        return query_axis >= 0 && p.input_shape(KernelInput::D, query_axis) != 1
            ? size_t(start) * p.input_stride(KernelInput::D, query_axis) : 0;
    }
    size_t head_mask_offset(const KernelParams& p, int head) const {
        int rank = p.input_rank(KernelInput::D);
        size_t result = 0;
        if (rank == 4 && p.input_shape(KernelInput::D, 0) != 1)
            result += size_t(head / heads) * p.input_stride(KernelInput::D, 0);
        if (rank >= 3 && p.input_shape(KernelInput::D, rank - 3) != 1)
            result += size_t(head % heads) * p.input_stride(KernelInput::D, rank - 3);
        return result;
    }
};
inline const uint32_t* attention_offset(const uint32_t* data, size_t count, size_t bytes) {
    return count ? reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(data) + count * bytes) : data;
}
inline uint32_t* attention_offset(uint32_t* data, size_t count, size_t bytes) {
    return count ? reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(data) + count * bytes) : data;
}
} // namespace elman
