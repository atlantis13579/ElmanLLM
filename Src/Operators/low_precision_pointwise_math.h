#pragma once
#include "Core/LowPrecisionFloat.h"
#include "Operators/low_precision_pointwise.h"
#include "Operators/tensor_axes.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_LOW_POINTWISE_HD __device__
#else
#define ELMAN_LOW_POINTWISE_HD
#endif
namespace elman {
ELMAN_LOW_POINTWISE_HD inline uint8_t low_precision_code(const uint32_t* data, int index, DType dtype) {
    const auto* bytes = reinterpret_cast<const uint8_t*>(data);
    return dtype == DType::Float4E2M1FN ? uint8_t((bytes[index / 2] >> ((index & 1) * 4)) & 15) : bytes[index];
}
ELMAN_LOW_POINTWISE_HD inline float low_precision_value(const uint32_t* data, int index, DType dtype) {
    if (dtype == DType::Float32) return reinterpret_cast<const float*>(data)[index];
    const uint8_t code = low_precision_code(data, index, dtype);
    if (dtype == DType::Bool) return float(code != 0);
    return dtype == DType::Float4E2M1FN ? float(numeric::float4_to_double(code)) : numeric::float8_to_float(code, dtype);
}
ELMAN_LOW_POINTWISE_HD inline uint8_t low_precision_encode(float value, DType dtype) {
    if (dtype == DType::Bool) return uint8_t(value != 0);
    if (is_float8(dtype)) return numeric::float_to_float8(value, dtype);
    const uint8_t code = numeric::double_to_float4(double(value));
    // The original Float32 sign also specifies the signed-zero NaN result.
    return uint8_t((code & 7) | ((numeric::low_float_bits(value) >> 28) & 8));
}
ELMAN_LOW_POINTWISE_HD inline float low_precision_pointwise_value(LowPrecisionPointwise op, float x, float y, float slope) {
    switch (op) {
    case LowPrecisionPointwise::Add: return x + y;
    case LowPrecisionPointwise::Sub: return x - y;
    case LowPrecisionPointwise::Mul: return x * y;
    case LowPrecisionPointwise::Div: return x / y;
    case LowPrecisionPointwise::Pow: return ::powf(x, y);
    case LowPrecisionPointwise::Maximum:
#ifdef __CUDACC__
        return x != x ? x : y != y ? y : ::fmaxf(x, y);
#else
        return x != x || x > y ? x : y;
#endif
    case LowPrecisionPointwise::Minimum:
#ifdef __CUDACC__
        return x != x ? x : y != y ? y : ::fminf(x, y);
#else
        return x != x || x < y ? x : y;
#endif
    case LowPrecisionPointwise::Eq: return x == y;
    case LowPrecisionPointwise::Ne: return x != y;
    case LowPrecisionPointwise::Gt: return x > y;
    case LowPrecisionPointwise::Ge: return x >= y;
    case LowPrecisionPointwise::Lt: return x < y;
    case LowPrecisionPointwise::Le: return x <= y;
    case LowPrecisionPointwise::LogicalAnd: return x != 0 && y != 0;
    case LowPrecisionPointwise::LogicalOr: return x != 0 || y != 0;
    case LowPrecisionPointwise::LogicalNot: return x == 0;
    case LowPrecisionPointwise::Sqrt:
#ifndef __CUDACC__
        // CPU sqrt and log emit positive NaNs for domain errors.
        if (x < 0) return numeric::low_bits_float(0x7fc00000);
#endif
        return ::sqrtf(x);
    case LowPrecisionPointwise::Rsqrt: return 1.f / ::sqrtf(x);
    case LowPrecisionPointwise::Exp: return ::expf(x);
    case LowPrecisionPointwise::Log:
#ifndef __CUDACC__
        if (x < 0) return numeric::low_bits_float(0x7fc00000);
#endif
        return ::logf(x);
    case LowPrecisionPointwise::Sin: return ::sinf(x);
    case LowPrecisionPointwise::Cos: return ::cosf(x);
    case LowPrecisionPointwise::Abs: return ::fabsf(x);
    case LowPrecisionPointwise::Sign: return float(x > 0) - float(x < 0);
    case LowPrecisionPointwise::Relu:
#ifdef __CUDACC__
        return x != x ? x : ::fmaxf(x, 0.f);
#else
        return x < 0 ? 0.f : x;
#endif
    case LowPrecisionPointwise::Silu: return x / (1.f + ::expf(-x));
    case LowPrecisionPointwise::Gelu: return .5f * x * (1.f + ::tanhf(.7978845608028654f * (x + .044715f * x * x * x)));
    case LowPrecisionPointwise::LeakyRelu: return x > 0 ? x : x * slope;
    default: return x;
    }
}
ELMAN_LOW_POINTWISE_HD inline uint8_t low_precision_pointwise_result(LowPrecisionPointwise op, int index,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int ai = broadcast_index(index, KernelInput::A, p);
    const auto dtype = DType(p.output_dtype());
    const float x = low_precision_value(a, ai, DType(p.input_dtype(KernelInput::A)));
#ifndef __CUDACC__
    // CPU FP8 abs preserves negative zero and unordered inputs.
    if (op == LowPrecisionPointwise::Abs && is_float8(dtype))
        return x < 0 ? low_precision_encode(-x, dtype) : low_precision_code(a, ai, dtype);
#endif
    if (op == LowPrecisionPointwise::Where) {
        const auto selected = x != 0 ? KernelInput::B : KernelInput::C;
        return low_precision_code(x != 0 ? b : c, broadcast_index(index, selected, p), dtype);
    }
    const float y = low_precision_pointwise_arity(op) == 2 ?
        low_precision_value(b, broadcast_index(index, KernelInput::B, p), DType(p.input_dtype(KernelInput::B))) : 0.f;
    return low_precision_encode(low_precision_pointwise_value(op, x, y, numeric::low_bits_float(uint32_t(p.attribute(0)))), dtype);
}
} // namespace elman
#undef ELMAN_LOW_POINTWISE_HD
