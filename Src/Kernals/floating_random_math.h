#pragma once
#include "Core/Float16.h"
#include "Kernals/float64_random_math.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#define ELMAN_FLOATING_RANDOM_HD __host__ __device__
#else
#define ELMAN_FLOATING_RANDOM_HD
#endif
namespace elman {
ELMAN_FLOATING_RANDOM_HD inline float floating_random_round(float value, DType dtype) {
#if defined(__CUDA_ARCH__)
    if (dtype == DType::Float16) return __half2float(__float2half_rn(value));
    if (dtype == DType::BFloat16) return __bfloat162float(__float2bfloat16_rn(value));
#else
    if (dtype == DType::Float16) return numeric::half_to_float(numeric::float_to_half(value));
    if (dtype == DType::BFloat16) return numeric::bfloat_to_float(numeric::float_to_bfloat(value));
#endif
    return value;
}
ELMAN_FLOATING_RANDOM_HD inline float floating_random_unit(uint64_t index, const KernelParams& p, DType dtype) {
    const double unit = float64_random_unit(index, p.attribute(0), p.attribute(1));
#if defined(__CUDA_ARCH__)
    return float(unit);
#else
    const int digits = dtype == DType::Float16 ? 11 : dtype == DType::BFloat16 ? 8 : 24;
    const uint32_t divisor = uint32_t(1) << digits;
    return float(uint32_t(unit * double(divisor))) / float(divisor);
#endif
}
ELMAN_FLOATING_RANDOM_HD inline float floating_random_value(bool normal, int index, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    const double first = float64_random_parameter(p, 0), second = float64_random_parameter(p, 1);
    if (!normal) {
        const float low = floating_random_round(float(first), dtype), high = floating_random_round(float(second), dtype);
        const float range = floating_random_round(high - low, dtype);
        float value = floating_random_round(::fmaf(range, floating_random_unit(uint64_t(index), p, dtype), low), dtype);
#if defined(__CUDA_ARCH__)
        // The native CUDA kernal restores the lower endpoint after storage rounding.
        if (value == high) value = low;
#endif
        return value;
    }
#if !defined(__CUDA_ARCH__)
    if (p.output_count() < 16)
        return floating_random_round(float(float64_random_value(Float64Random::Normal, uint64_t(index), 0., p)), dtype);
    if (dtype != DType::Float32) {
        // The contiguous CPU reduced-format path stores every intermediate in
        // the tensor dtype, including its uniform samples and transcendental results.
        const uint64_t pair = uint64_t(index / 16) * 8 + uint64_t(index % 8);
        const float u1 = floating_random_round(1.f - floating_random_unit(2 * pair, p, dtype), dtype);
        const float u2 = floating_random_unit(2 * pair + 1, p, dtype);
        const float logarithm = floating_random_round(::logf(u1), dtype);
        const float radius = floating_random_round(::sqrtf(floating_random_round(-2.f * logarithm, dtype)), dtype);
        const float theta = floating_random_round(float(6.283185307179586476925286766559 * double(u2)), dtype);
        const float trigonometric = floating_random_round(index % 16 < 8 ? ::cosf(theta) : ::sinf(theta), dtype);
        const float product = floating_random_round(radius * trigonometric, dtype);
        const float scaled = floating_random_round(product * floating_random_round(float(second), dtype), dtype);
        return floating_random_round(scaled + floating_random_round(float(first), dtype), dtype);
    }
#endif
    const uint64_t counter = uint64_t(index) * 2;
    const float u = float(1. - float64_random_unit(counter, p.attribute(0), p.attribute(1)));
    const float v = float(float64_random_unit(counter + 1, p.attribute(0), p.attribute(1)));
    const float sample = ::sqrtf(-2.f * ::logf(u)) * ::cosf(6.283185307179586f * v);
    return floating_random_round(::fmaf(float(second), sample, float(first)), dtype);
}
} // namespace elman
#undef ELMAN_FLOATING_RANDOM_HD
