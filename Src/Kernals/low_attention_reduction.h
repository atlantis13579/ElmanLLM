#pragma once
#include "Core/Backend.h"

#ifdef __CUDACC__
#define ELMAN_ATTENTION_HD __host__ __device__
#else
#define ELMAN_ATTENTION_HD
#endif
namespace elman {
struct AttentionReduction {
    // Collapse adjacent reduced axes while retaining physical input strides.
    // CPU cascade sums and CUDA lane layout both depend on those boundaries.
    int shape[8]{}, strides[8]{}, rank = 0, mask = 0, length = 1, rows = 1;
    int block = 1, inner = 1, reduced_blocks = 0;
    bool fastest = false;
    explicit AttentionReduction(const KernelParams& p) {
        rank = p.input_rank(KernelInput::A); mask = p.attribute(8) == 3 ? 2 : p.attribute(0);
        bool previous = false, found = false, block_done = false;
        for (int d = rank - 1; d >= 0; --d) {
            shape[d] = p.input_shape(KernelInput::A, d); strides[d] = p.input_stride(KernelInput::A, d);
            const bool reduced = (mask & (1 << d)) != 0;
            if (reduced) length *= shape[d]; else rows *= shape[d];
            if (shape[d] == 1) continue;
            if (!found) { fastest = reduced; found = true; }
            if (reduced && !previous) ++reduced_blocks;
            if (!reduced && !reduced_blocks) inner *= shape[d];
            if (reduced && !block_done) block *= shape[d];
            if (!reduced && reduced_blocks) block_done = true;
            previous = reduced;
        }
    }
    ELMAN_ATTENTION_HD int offset(int output, int reduction) const {
        int result = 0;
        for (int d = rank - 1; d >= 0; --d) {
            if (mask & (1 << d)) { result += reduction % shape[d] * strides[d]; reduction /= shape[d]; }
            else { result += output % shape[d] * strides[d]; output /= shape[d]; }
        }
        return result;
    }
};
} // namespace elman
#undef ELMAN_ATTENTION_HD
