#pragma once
#include "Operators/float64_matrix.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_MATRIX_HD __host__ __device__
#else
#define ELMAN_DOUBLE_MATRIX_HD
#endif

namespace elman {
ELMAN_DOUBLE_MATRIX_HD inline int float64_matrix_batch(int batch, KernelInput input, const KernelParams& p) {
    int offset = 0, delta = p.output_rank() - p.input_rank(input);
    for (int d = p.output_rank() - 3; d >= 0; --d) {
        const int coordinate = batch % p.output_shape(d);
        batch /= p.output_shape(d);
        if (d >= delta && p.input_shape(input, d - delta) != 1)
            offset += coordinate * p.input_stride(input, d - delta);
    }
    return offset;
}
ELMAN_DOUBLE_MATRIX_HD inline double float64_matrix_coefficient(const KernelParams& p, int slot) {
    union { uint64_t bits; double value; } scalar;
    scalar.bits = uint64_t(uint32_t(p.attribute(slot))) | (uint64_t(uint32_t(p.attribute(slot + 1))) << 32);
    return scalar.value;
}
ELMAN_DOUBLE_MATRIX_HD inline double float64_matrix_value(Float64Matrix operation, int index,
        const double* a, const double* b, const double* c, const KernelParams& p) {
    const bool linear = operation == Float64Matrix::Linear || operation == Float64Matrix::PackedLinear;
    const bool addmm = operation == Float64Matrix::Addmm;
    int arow, bstart, bstride, shared, column;
    if (linear) {
        const int columns = p.input_shape(KernelInput::B, 0);
        shared = p.input_shape(KernelInput::B, 1);
        int row = index / columns;
        column = index % columns;
        if (operation == Float64Matrix::PackedLinear) {
            const int rows = p.output_count() / columns;
            int remaining = index, begin = 0;
            for (int group = 0; group < p.attribute(0); ++group) {
                const int width = p.attribute(group + 1);
                if (remaining < rows * width) {
                    row = remaining / width;
                    column = begin + remaining % width;
                    break;
                }
                remaining -= rows * width;
                begin += width;
            }
        }
        arow = row * shared;
        bstart = column * shared;
        bstride = 1;
    } else {
        const int columns = p.output_shape(p.output_rank() - 1), rows = p.output_shape(p.output_rank() - 2);
        const int row = index / columns % rows, batch = index / (rows * columns);
        column = index % columns;
        shared = p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - 1);
        const bool transposed = operation == Float64Matrix::Matmul && p.attribute(0) != 0;
        arow = float64_matrix_batch(batch, KernelInput::A, p) + row * shared;
        bstart = float64_matrix_batch(batch, KernelInput::B, p) + (transposed ? column * shared : column);
        bstride = transposed ? 1 : columns;
    }
    const double alpha = addmm ? float64_matrix_coefficient(p, 0) : 1.;
    double value = 0.;
    // Zero alpha suppresses matrix reads, and an empty contraction suppresses
    // alpha itself, including nonfinite coefficients.
    for (int k = 0; alpha != 0. && k < shared; ++k)
        value = ::fma(a[arow + k], b[bstart + k * bstride], value);
    if (addmm) {
        if (shared && alpha != 0.) value *= alpha;
        const double beta = float64_matrix_coefficient(p, 2);
        if (beta != 0.) {
            int addend = 0, flat = index, delta = p.output_rank() - p.input_rank(KernelInput::C);
            for (int d = p.output_rank() - 1; d >= delta; --d) {
                const int coordinate = flat % p.output_shape(d);
                flat /= p.output_shape(d);
                if (p.input_shape(KernelInput::C, d - delta) != 1)
                    addend += coordinate * p.input_stride(KernelInput::C, d - delta);
            }
            value += beta * c[addend];
        }
    } else if (linear && p.input_rank(KernelInput::C) >= 0) {
        value += c[p.input_rank(KernelInput::C) == 0 ? 0 : column];
    }
    return value;
}
} // namespace elman
#undef ELMAN_DOUBLE_MATRIX_HD
