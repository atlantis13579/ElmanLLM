#pragma once
#include "Operators/float64_random.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_RANDOM_HD __host__ __device__
#else
#define ELMAN_DOUBLE_RANDOM_HD
#endif
namespace elman {
// Slots 4-8 extend the original two Float32 parameters without changing them.
ELMAN_DOUBLE_RANDOM_HD inline double float64_random_parameter(const KernelParams& p, int parameter) {
    if (p.attribute(8)) {
        union { uint64_t bits; double value; } scalar;
        scalar.bits = uint64_t(uint32_t(p.attribute(4 + 2 * parameter))) |
            (uint64_t(uint32_t(p.attribute(5 + 2 * parameter))) << 32);
        return scalar.value;
    }
    union { uint32_t bits; float value; } scalar;
    scalar.bits = uint32_t(p.attribute(2 + parameter));
    return double(scalar.value);
}
ELMAN_DOUBLE_RANDOM_HD inline double float64_random_unit(uint64_t index, int seed, int offset) {
    // A stateless 64-bit counter mixer makes replay independent of launch shape.
    uint64_t value = index + uint64_t(uint32_t(seed)) * 0x9e3779b97f4a7c15ull +
        uint64_t(uint32_t(offset)) * 0xd1b54a32d192ed03ull + 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    value ^= value >> 31;
    return double(value >> 11) * (1. / 9007199254740992.);
}
ELMAN_DOUBLE_RANDOM_HD inline double float64_random_value(Float64Random op, uint64_t index, double input, const KernelParams& p) {
    const double first = float64_random_parameter(p, 0);
    if (op == Float64Random::Dropout) {
        if (!p.attribute(3) || first == 0.) return input;
        const bool keep = first < 1. && float64_random_unit(index, p.attribute(0), p.attribute(1)) >= first;
        // Multiplication preserves dropped NaN/Inf and the sign of zero.
        return input * (keep ? 1. / (1. - first) : 0.);
    }
    const double second = float64_random_parameter(p, 1);
    if (op == Float64Random::Uniform) {
        double result = ::fma(second - first, float64_random_unit(index, p.attribute(0), p.attribute(1)), first);
#if defined(__CUDA_ARCH__)
        // CUDA's reference maps a rounded upper endpoint back to the lower one.
        if (result == second) result = first;
#endif
        return result;
    }
    const double u = 1. - float64_random_unit(2 * index, p.attribute(0), p.attribute(1));
    const double v = float64_random_unit(2 * index + 1, p.attribute(0), p.attribute(1));
    const double normal = ::sqrt(-2. * ::log(u)) * ::cos(6.283185307179586476925286766559 * v);
    return ::fma(second, normal, first);
}
} // namespace elman
#undef ELMAN_DOUBLE_RANDOM_HD
