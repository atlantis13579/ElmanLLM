#pragma once
#include "Kernals/low_precision_fused.h"
#include "Kernals/low_precision_pointwise_math.h"
#ifdef __CUDACC__
#define ELMAN_LOW_FUSED_HD __device__
#else
#define ELMAN_LOW_FUSED_HD
#endif
namespace elman {
ELMAN_LOW_FUSED_HD inline float low_fused_multiply(float a, float b) {
#ifdef __CUDACC__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
ELMAN_LOW_FUSED_HD inline float low_precision_fused_value(LowPrecisionFused op, int i, const uint32_t* a,
                                                          const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    const float x = low_precision_value(a, broadcast_index(i, KernelInput::A, p), dtype);
    const float y = low_precision_value(b, broadcast_index(i, KernelInput::B, p), dtype);
    if (op == LowPrecisionFused::SiluMul) return low_fused_multiply(low_precision_pointwise_value(LowPrecisionPointwise::Silu, x, 0.f, 0.f), y);
    const int width = p.output_shape(p.output_rank() - 1), half = width / 2;
    const bool first = i % width < half;
    const float paired = low_precision_value(a, first ? i + half : i - half, dtype);
    const float rotated = first ? -paired : paired;
    const float sine = low_precision_value(c, broadcast_index(i, KernelInput::C, p), dtype);
    const float left = low_fused_multiply(x, y), right = low_fused_multiply(rotated, sine);
#ifdef __CUDACC__
    return __fadd_rn(left, right);
#else
    return left + right;
#endif
}
} // namespace elman
#undef ELMAN_LOW_FUSED_HD
