#pragma once
#include "Core/Backend.h"
#include "Backends/AttentionGemm.h"
#include "Operators/attributes.h"
#include <algorithm>

namespace elman {
void validate_paged_attention(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& params);

struct PagedAttentionShape {
    int batch, heads, queries, dimension, keys, kv_heads, slots, tile, dtype;
    explicit PagedAttentionShape(const KernelParams& p)
        : batch(p.input_shape(KernelInput::A, 0)), heads(p.input_shape(KernelInput::A, 1)),
          queries(p.input_shape(KernelInput::A, 2)), dimension(p.input_shape(KernelInput::A, 3)),
          keys(p.input_shape(KernelInput::D, 1)), kv_heads(p.attribute(4)),
          slots(p.input_shape(KernelInput::B, 0)), tile(std::min(queries, p.attribute(1))), dtype(p.output_dtype()) {}
    KernelParams probabilities(const KernelParams& source, int rows, int start, bool one_head = false) const {
        KernelParams p;
        const int shape[] = {one_head ? 1 : batch, one_head ? 1 : heads, rows, keys};
        const int mask[] = {one_head ? 1 : batch, 1, 1, keys};
        p.v[KernelParams::kOutputCountSlot] = shape[0] * shape[1] * rows * keys;
        p.v[KernelParams::kOutputRankSlot] = 4;
        p.v[KernelParams::kOutputDTypeSlot] = dtype;
        for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
        for (int input = 0; input < 2; ++input) {
            const KernelInput id = KernelInput(input);
            const int base = KernelParams::input_slot(id);
            p.v[base] = 4;
            p.v[KernelParams::input_dtype_slot(id)] = dtype;
            int stride = 1;
            for (int d = 3; d >= 0; --d) {
                const int length = input ? mask[d] : shape[d];
                p.v[base + KernelParams::kInputShapeOffset + d] = length;
                p.v[base + KernelParams::kInputStrideOffset + d] = stride;
                stride *= length;
                p.v[KernelParams::kOutputShapeSlot + d] = shape[d];
            }
        }
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kAxis] = 3;
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kDivisor] = source.attribute(0);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kCausal] = source.attribute(2);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kCausalOffset] = source.attribute(3);
        p.v[KernelParams::kAttributeSlot + operator_attr::attention_softmax::kQueryBase] = start;
        return p;
    }
};
} // namespace elman
