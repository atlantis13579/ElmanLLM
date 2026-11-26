#pragma once
#include "Kernals/low_precision_reduction.h"
#include "Kernals/low_precision_pointwise_math.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_LOW_REDUCTION_HD __device__
#else
#define ELMAN_LOW_REDUCTION_HD
#endif
namespace elman {
ELMAN_LOW_REDUCTION_HD inline int low_precision_reduction_length(const KernelParams& p) {
    if (p.attribute(0) >= 0) return p.input_rank(KernelInput::A) == 0 ? 1 : p.input_shape(KernelInput::A, p.attribute(0));
    int count = 1;
    for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
    return count;
}
ELMAN_LOW_REDUCTION_HD inline int low_precision_reduction_inner(const KernelParams& p) {
    return p.attribute(0) < 0 || p.input_rank(KernelInput::A) == 0 ? 1 : p.input_stride(KernelInput::A, p.attribute(0));
}
ELMAN_LOW_REDUCTION_HD inline float low_precision_reduce_row(LowPrecisionReduction op, const uint32_t* input,
                                                     int base, int length, int inner, DType dtype) {
    if (op == LowPrecisionReduction::Amax) op = LowPrecisionReduction::Maximum;
    if (op == LowPrecisionReduction::Amin) op = LowPrecisionReduction::Minimum;
    float value = op == LowPrecisionReduction::Product ? 1. : op == LowPrecisionReduction::Maximum ? -INFINITY :
        op == LowPrecisionReduction::Minimum ? INFINITY : 0.;
    for (int j = 0; j < length; ++j) {
        const float x = low_precision_value(input, base + j * inner, dtype);
        if (op == LowPrecisionReduction::Product) value *= x;
        else if (op == LowPrecisionReduction::Maximum || op == LowPrecisionReduction::Minimum) {
            // Select through integer representations so host min/max instructions
            // cannot replace the first operand's zero sign on equal values.
            const bool retain = value != value || (op == LowPrecisionReduction::Maximum ? value >= x : value <= x);
            const uint32_t bits = retain ? numeric::low_float_bits(value) : numeric::low_float_bits(x);
            value = numeric::low_bits_float(bits);
        }
        else value += x;
    }
    return op == LowPrecisionReduction::Mean ? value / float(length) : value;
}
ELMAN_LOW_REDUCTION_HD inline float low_precision_reduce_gradient(int i, const uint32_t* input, const uint32_t* saved,
                                                         const uint32_t* upstream, const KernelParams& p) {
    const DType dtype = DType(p.input_dtype(KernelInput::A));
    const int length = low_precision_reduction_length(p), inner = low_precision_reduction_inner(p);
    const int target = p.attribute(0) < 0 ? 0 : (i / (length * inner)) * inner + i % inner;
    const int base = p.attribute(0) < 0 ? 0 : (target / inner) * length * inner + target % inner;
    float value = low_precision_value(upstream, target, DType(p.input_dtype(KernelInput::C)));
    const int kind = p.attribute(2);
    if (kind == 1) return value / float(length);
    if (kind == 2 || kind == 3) {
        // Scalar amax/amin backward reduces the restored gradient to a scalar.
        if (p.input_rank(KernelInput::A) == 0 && value == 0.f) return 0.f;
        int matches = 0;
        for (int j = 0; j < length; ++j) matches += low_precision_value(input, base + j * inner, dtype) == low_precision_value(saved, target, DType(p.input_dtype(KernelInput::B)));
        return (value / float(matches)) * float(low_precision_value(input, i, dtype) == low_precision_value(saved, target, DType(p.input_dtype(KernelInput::B))));
    }
    if (kind == 5 || kind == 6) {
        for (int j = 0; j < length; ++j) {
            const int index = base + j * inner;
            if (low_precision_value(input, index, dtype) == low_precision_value(saved, target, DType(p.input_dtype(KernelInput::B))) || (low_precision_value(input, index, dtype) != low_precision_value(input, index, dtype) && low_precision_value(saved, target, DType(p.input_dtype(KernelInput::B))) != low_precision_value(saved, target, DType(p.input_dtype(KernelInput::B)))))
                return index == i ? value : 0.;
        }
        return 0.;
    }
    if (kind == 4) {
        // The reference switches the entire tensor to exclusive products when
        // any input is zero, including rows containing NaNs or infinities.
        int count = 1;
        for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
        bool zero = false;
        for (int j = 0; j < count && !zero; ++j) zero = low_precision_value(input, j, dtype) == 0.;
        if (!zero) return (value * low_precision_reduce_row(LowPrecisionReduction::Product, input, base, length, inner, dtype)) / low_precision_value(input, i, dtype);
        const int position = (i - base) / inner;
        float left = 1., right = 1.;
        for (int j = 0; j < position; ++j) left *= low_precision_value(input, base + j * inner, dtype);
        for (int j = length - 1; j > position; --j) right *= low_precision_value(input, base + j * inner, dtype);
        return value * (left * right);
    }
    return value;
}
ELMAN_LOW_REDUCTION_HD inline float low_precision_reduction_result(LowPrecisionReduction op, int i,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (op == LowPrecisionReduction::Gradient) return low_precision_reduce_gradient(i, a, b, c, p);
    const int length = low_precision_reduction_length(p), inner = low_precision_reduction_inner(p);
    const int base = p.attribute(0) < 0 ? 0 : i / inner * length * inner + i % inner;
    return low_precision_reduce_row(op, a, base, length, inner, DType(p.input_dtype(KernelInput::A)));
}
} // namespace elman
#undef ELMAN_LOW_REDUCTION_HD
