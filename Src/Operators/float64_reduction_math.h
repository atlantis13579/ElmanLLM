#pragma once
#include "Operators/float64_reduction.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_REDUCTION_HD __host__ __device__
#else
#define ELMAN_REDUCTION_HD
#endif
namespace elman {
ELMAN_REDUCTION_HD inline int float64_reduction_length(const KernelParams& p) {
    if (p.attribute(0) >= 0) return p.input_rank(KernelInput::A) == 0 ? 1 : p.input_shape(KernelInput::A, p.attribute(0));
    int count = 1;
    for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
    return count;
}
ELMAN_REDUCTION_HD inline int float64_reduction_inner(const KernelParams& p) {
    return p.attribute(0) < 0 || p.input_rank(KernelInput::A) == 0 ? 1 : p.input_stride(KernelInput::A, p.attribute(0));
}
ELMAN_REDUCTION_HD inline double float64_reduce_row(Float64Reduction op, const double* input,
                                                     int base, int length, int inner) {
    double value = op == Float64Reduction::Product ? 1. : op == Float64Reduction::Maximum ? -INFINITY :
        op == Float64Reduction::Minimum ? INFINITY : 0.;
    for (int j = 0; j < length; ++j) {
        const double x = input[base + j * inner];
        if (op == Float64Reduction::Product) value *= x;
        else if (op == Float64Reduction::Maximum) value = value != value || value >= x ? value : x;
        else if (op == Float64Reduction::Minimum) value = value != value || value <= x ? value : x;
        else value += x;
    }
    return op == Float64Reduction::Mean ? value / double(length) : value;
}
ELMAN_REDUCTION_HD inline double float64_reduce_gradient(int i, const double* input, const double* saved,
                                                         const double* upstream, const KernelParams& p) {
    const int length = float64_reduction_length(p), inner = float64_reduction_inner(p);
    const int target = p.attribute(0) < 0 ? 0 : (i / (length * inner)) * inner + i % inner;
    const int base = p.attribute(0) < 0 ? 0 : (target / inner) * length * inner + target % inner;
    double value = upstream[target];
    const int kind = p.attribute(2);
    if (kind == 1) return value / double(length);
    if (kind == 2 || kind == 3) {
        int matches = 0;
        for (int j = 0; j < length; ++j) matches += input[base + j * inner] == saved[target];
        return (value / double(matches)) * double(input[i] == saved[target]);
    }
    if (kind == 5 || kind == 6) {
        for (int j = 0; j < length; ++j) {
            const int index = base + j * inner;
            if (input[index] == saved[target] || (input[index] != input[index] && saved[target] != saved[target]))
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
        for (int j = 0; j < count && !zero; ++j) zero = input[j] == 0.;
        if (!zero) return (value * saved[target]) / input[i];
        const int position = (i - base) / inner;
        double left = 1., right = 1.;
        for (int j = 0; j < position; ++j) left *= input[base + j * inner];
        for (int j = length - 1; j > position; --j) right *= input[base + j * inner];
        return value * (left * right);
    }
    return value;
}
ELMAN_REDUCTION_HD inline void float64_scan_row(double* output, const double* input, int row,
                                                int length, int inner, bool reverse) {
    const int base = row / inner * length * inner + row % inner;
    double value = 0.;
    for (int j = 0; j < length; ++j) {
        const int index = base + (reverse ? length - 1 - j : j) * inner;
        value += input[index];
        output[index] = value;
    }
}
} // namespace elman
#undef ELMAN_REDUCTION_HD
