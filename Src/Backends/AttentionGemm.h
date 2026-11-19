#pragma once
#include "Core/Backend.h"

namespace elman {
// A and C retain their complete query strides while processing one row tile.
// Consecutive groups of query heads share the same right-hand matrix.
struct AttentionGemm {
    DType dtype;
    int rows, columns, shared, batches, groups;
    int a_stride, b_stride, output_stride;
    bool transpose_b;
    bool serial_cpu = false;
    int b_leading = 0;
};
bool try_attention_gemm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const AttentionGemm& shape);
void attention_gemm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         void* pointers, const AttentionGemm& shape);
void attention_gemm_cuda_pointers(void* pointers, const AttentionGemm& shape);
} // namespace elman
