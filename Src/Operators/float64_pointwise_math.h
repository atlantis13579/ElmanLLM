#pragma once
#include "Operators/float64_pointwise.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_HD __host__ __device__
#else
#define ELMAN_DOUBLE_HD
#endif
namespace elman {
ELMAN_DOUBLE_HD inline double float64_pointwise_slope(const KernelParams& p) {
    if (p.attribute(2)) {
        union { uint64_t bits; double value; } scalar;
        scalar.bits = uint64_t(uint32_t(p.attribute(0))) | (uint64_t(uint32_t(p.attribute(1))) << 32);
        return scalar.value;
    }
    union { uint32_t bits; float value; } scalar;
    scalar.bits = uint32_t(p.attribute(0));
    return double(scalar.value);
}
ELMAN_DOUBLE_HD inline double float64_rotary_value(double x, double cosine, double rotated, double sine) {
    // Preserve the separate product rounding of the eager reference.
#if defined(__CUDA_ARCH__)
    return __dadd_rn(__dmul_rn(x, cosine), __dmul_rn(rotated, sine));
#else
    const double left = x * cosine, right = rotated * sine;
    return left + right;
#endif
}
ELMAN_DOUBLE_HD inline double float64_pointwise_value(Float64Pointwise op, double x, double y,
                                                      double z, double slope) {
    switch (op) {
    case Float64Pointwise::Cast: return x;
    case Float64Pointwise::Add: return x + y;
    case Float64Pointwise::Sub: return x - y;
    case Float64Pointwise::Mul: return x * y;
    case Float64Pointwise::Div: return x / y;
    case Float64Pointwise::Pow: return ::pow(x, y);
    case Float64Pointwise::Maximum: return x != x || x > y ? x : y;
    case Float64Pointwise::Minimum: return x != x || x < y ? x : y;
    case Float64Pointwise::Eq: return x == y;
    case Float64Pointwise::Ne: return x != y;
    case Float64Pointwise::Gt: return x > y;
    case Float64Pointwise::Ge: return x >= y;
    case Float64Pointwise::Lt: return x < y;
    case Float64Pointwise::Le: return x <= y;
    case Float64Pointwise::LogicalAnd: return x != 0 && y != 0;
    case Float64Pointwise::LogicalOr: return x != 0 || y != 0;
    case Float64Pointwise::LogicalNot: return x == 0;
    case Float64Pointwise::Sqrt: return ::sqrt(x);
    case Float64Pointwise::Rsqrt: return 1. / ::sqrt(x);
    case Float64Pointwise::Exp: return ::exp(x);
    case Float64Pointwise::Log: return ::log(x);
    case Float64Pointwise::Sin: return ::sin(x);
    case Float64Pointwise::Cos: return ::cos(x);
    case Float64Pointwise::Abs: return ::fabs(x);
    case Float64Pointwise::Sign: return double(x > 0) - double(x < 0);
    case Float64Pointwise::Relu: return x < 0 ? 0. : x;
    case Float64Pointwise::Silu: return x / (1. + ::exp(-x));
    case Float64Pointwise::Gelu: return .5 * x * (1. + ::tanh(.79788456080286535588 * (x + .044715 * x * x * x)));
    case Float64Pointwise::LeakyRelu: return x > 0 ? x : x * slope;
    case Float64Pointwise::Where: return x != 0 ? y : z;
    case Float64Pointwise::Fma: return ::fma(x, y, z);
    case Float64Pointwise::SiluMul: return (x / (1. + ::exp(-x))) * y;
    default: return x;
    }
}
} // namespace elman
#undef ELMAN_DOUBLE_HD
