#pragma once
#include "Kernals/low_precision_matrix.h"
#include "Kernals/low_precision_pointwise_math.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_LOW_MATRIX_HD __device__
#else
#define ELMAN_LOW_MATRIX_HD
#endif

namespace elman {
ELMAN_LOW_MATRIX_HD inline int low_precision_matrix_batch(int batch, KernelInput input, const KernelParams& p) {
    int offset = 0, delta = p.output_rank() - p.input_rank(input);
    for (int d = p.output_rank() - 3; d >= 0; --d) {
        const int coordinate = batch % p.output_shape(d);
        batch /= p.output_shape(d);
        if (d >= delta && p.input_shape(input, d - delta) != 1)
            offset += coordinate * p.input_stride(input, d - delta);
    }
    return offset;
}
ELMAN_LOW_MATRIX_HD inline float low_precision_matrix_coefficient(const KernelParams& p, int slot) {
    return numeric::low_bits_float(uint32_t(p.attribute(slot)));
}
ELMAN_LOW_MATRIX_HD inline float low_precision_matrix_value(LowPrecisionMatrix operation, int index,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const bool linear = operation == LowPrecisionMatrix::Linear || operation == LowPrecisionMatrix::PackedLinear;
    const bool addmm = operation == LowPrecisionMatrix::Addmm;
    int arow, bstart, bstride, shared, column;
    if (linear) {
        const int columns = p.input_shape(KernelInput::B, 0);
        shared = p.input_shape(KernelInput::B, 1);
        int row = index / columns;
        column = index % columns;
        if (operation == LowPrecisionMatrix::PackedLinear) {
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
        const bool transposed = operation == LowPrecisionMatrix::Matmul && p.attribute(0) != 0;
        arow = low_precision_matrix_batch(batch, KernelInput::A, p) + row * shared;
        bstart = low_precision_matrix_batch(batch, KernelInput::B, p) + (transposed ? column * shared : column);
        bstride = transposed ? 1 : columns;
    }
    const float alpha = addmm ? low_precision_matrix_coefficient(p, 0) : 1.;
    float value = 0.;
    bool read_product = alpha != 0.f;
#ifndef __CUDACC__
    // Native CPU FP8 GEMM applies alpha after accumulating even when it is zero.
    if (is_float8(DType(p.output_dtype())) && DType(p.output_dtype()) != DType::Float8E8M0FNU) read_product = true;
#endif
    bool outer_product = shared == 1;
#ifndef __CUDACC__
    const bool native_float8 = is_float8(DType(p.output_dtype())) && DType(p.output_dtype()) != DType::Float8E8M0FNU;
    const int columns = linear ? p.input_shape(KernelInput::B, 0) : p.output_shape(p.output_rank() - 1);
    const int rows = columns ? p.output_count() / columns : 0;
    outer_product = outer_product && !native_float8 && rows > 1 && columns > 1 &&
        (linear || addmm || p.output_rank() == 2 || (operation == LowPrecisionMatrix::Matmul && p.attribute(1) == 1));
#endif
    if (read_product && outer_product) {
        value = low_precision_value(a, arow, DType(p.input_dtype(KernelInput::A))) *
                low_precision_value(b, bstart, DType(p.input_dtype(KernelInput::B)));
    } else for (int k = 0; read_product && k < shared; ++k)
        value = ::fmaf(low_precision_value(a, arow + k, DType(p.input_dtype(KernelInput::A))), low_precision_value(b, bstart + k * bstride, DType(p.input_dtype(KernelInput::B))), value);
    if (addmm) {
        if (shared && read_product) value *= alpha;
        const float coefficient = low_precision_matrix_coefficient(p, 1);
        float beta = coefficient;
#ifndef __CUDACC__
        // The native empty CPU FP8 product scales its addend in the stored dtype.
        if (!shared && native_float8) beta = numeric::float8_to_float(numeric::float_to_float8(beta, DType(p.output_dtype())), DType(p.output_dtype()));
#endif
        if (coefficient != 0.) {
            int addend = 0, flat = index, delta = p.output_rank() - p.input_rank(KernelInput::C);
            for (int d = p.output_rank() - 1; d >= delta; --d) {
                const int coordinate = flat % p.output_shape(d);
                flat /= p.output_shape(d);
                if (p.input_shape(KernelInput::C, d - delta) != 1)
                    addend += coordinate * p.input_stride(KernelInput::C, d - delta);
            }
            const float addend_value = beta * low_precision_value(c, addend, DType(p.input_dtype(KernelInput::C)));
            bool add_product = shared && read_product;
#ifdef __CUDACC__
            // The vector-bias FP32 CUDA epilogue still adds the zero product.
            if (shared > 1 && p.output_shape(p.output_rank() - 1) > 1 && p.input_rank(KernelInput::C) == 1 && coefficient == 1.f)
                add_product = true;
#endif
            value = add_product ? value + addend_value : addend_value;
        }
    } else if (linear && p.input_rank(KernelInput::C) >= 0) {
        const float bias = low_precision_value(c, p.input_rank(KernelInput::C) == 0 ? 0 : column, DType(p.input_dtype(KernelInput::C)));
        value = shared ? value + bias : bias;
    }
    return value;
}
} // namespace elman
#undef ELMAN_LOW_MATRIX_HD
