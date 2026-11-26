#pragma once
#include "Kernals/low_precision_random.h"
#include "Kernals/low_precision_pointwise_math.h"
#include "Kernals/float64_random_math.h"
#include "Kernals/dropout_math.h"

#ifdef __CUDACC__
#define ELMAN_LOW_RANDOM_HD __device__
#else
#define ELMAN_LOW_RANDOM_HD
#endif
namespace elman {
ELMAN_LOW_RANDOM_HD inline uint8_t low_precision_random_code(LowPrecisionRandom op, int index, const uint32_t* input, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    const double first = float64_random_parameter(p, 0), second = float64_random_parameter(p, 1);
    float value;
    if (op == LowPrecisionRandom::Dropout) {
        if (!p.attribute(3) || first == 0.) return low_precision_code(input, index, dtype);
        // Reuse the counter stream across storage formats. The keep mask stays
        // boolean even when the output format cannot represent zero.
        const bool keep = first < 1. && float64_random_unit(uint64_t(index), p.attribute(0), p.attribute(1)) >= first;
        const float scale = floating_dropout_scale(DType::Float32, first, p.attribute(operator_attr::dropout::kBackward) != 0);
        value = low_precision_value(input, index, dtype) * (keep ? scale : 0.f);
    } else if (op == LowPrecisionRandom::Uniform) {
        const float low = float(first), high = float(second);
        const double counter = float64_random_unit(uint64_t(index), p.attribute(0), p.attribute(1));
#if defined(__CUDA_ARCH__)
        const float unit = float(counter);
#else
        // Match the CPU FP32 distribution's 24-bit unit grid. Rounding a wider
        // unit to FP32 could introduce an upper-endpoint value absent there.
        const float unit = float(uint32_t(counter * 16777216.)) * (1.f / 16777216.f);
#endif
        value = ::fmaf(high - low, unit, low);
#if defined(__CUDA_ARCH__)
        // CUDA also restores the lower endpoint's zero sign for equal bounds.
        if (value == high) value = low;
#endif
    } else {
#ifndef __CUDACC__
        if (p.output_count() < 16) {
            // The reference's short CPU normal path transforms in Double before
            // storing FP32. Preserve overflow classes and underflow zero signs.
            value = float(float64_random_value(Float64Random::Normal, uint64_t(index), 0., p));
        } else
#endif
        {
            const uint64_t counter = uint64_t(index) * 2;
            const float u = float(1. - float64_random_unit(counter, p.attribute(0), p.attribute(1)));
            const float v = float(float64_random_unit(counter + 1, p.attribute(0), p.attribute(1)));
            const float normal = ::sqrtf(-2.f * ::logf(u)) * ::cosf(6.283185307179586f * v);
            value = ::fmaf(float(second), normal, float(first));
        }
    }
    return low_precision_encode(value, dtype);
}
} // namespace elman
#undef ELMAN_LOW_RANDOM_HD
