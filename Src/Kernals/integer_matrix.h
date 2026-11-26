#pragma once
#include "Kernals/integer_pointwise.h"

#ifdef __CUDACC__
#define ELMAN_MATRIX_HD __host__ __device__
#else
#define ELMAN_MATRIX_HD
#endif

namespace elman {
enum class IntegerMatrix { None, Matmul, Bmm, Linear, PackedLinear, Addmm };

inline IntegerMatrix integer_matrix_operation(const std::string& name, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype());
    if (!is_standard_integer(dtype) || dtype == DType::Bool) return IntegerMatrix::None;
    IntegerMatrix operation = IntegerMatrix::None;
    if (name == "matmul") operation = IntegerMatrix::Matmul;
    else if (name == "bmm") operation = IntegerMatrix::Bmm;
    else if (name == "linear") operation = IntegerMatrix::Linear;
    else if (name == "packed_linear") operation = IntegerMatrix::PackedLinear;
    else if (name == "addmm") operation = IntegerMatrix::Addmm;
    // Keep the established mixed-dtype Int32 raw ABI on its original kernels.
    if (dtype == DType::Int32) for (KernelInput input : {KernelInput::A, KernelInput::B, KernelInput::C})
        if (p.input_rank(input) >= 0 && p.input_dtype(input) != int(dtype)) return IntegerMatrix::None;
    return operation;
}

ELMAN_MATRIX_HD inline int integer_matrix_batch(int batch, KernelInput input, const KernelParams& p) {
    int offset = 0, delta = p.output_rank() - p.input_rank(input);
    for (int d = p.output_rank() - 3; d >= 0; --d) {
        const int coordinate = batch % p.output_shape(d);
        batch /= p.output_shape(d);
        if (d >= delta && p.input_shape(input, d - delta) != 1)
            offset += coordinate * p.input_stride(input, d - delta);
    }
    return offset;
}
ELMAN_MATRIX_HD inline uint64_t integer_matrix_coefficient(const KernelParams& p, int slot) {
    union { int32_t bits; float value; } scalar;
    scalar.bits = p.attribute(slot);
    // Host descriptor validation establishes a finite, representable conversion.
    return integer_signed(p.output_dtype()) ? uint64_t(int64_t(scalar.value)) : uint64_t(scalar.value);
}
ELMAN_MATRIX_HD inline uint64_t integer_matrix_value(IntegerMatrix operation, int index,
        const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int dtype = p.output_dtype();
    const bool linear = operation == IntegerMatrix::Linear || operation == IntegerMatrix::PackedLinear;
    int arow, bstart, bstride, shared, column;
    if (linear) {
        const int columns = p.input_shape(KernelInput::B, 0);
        shared = p.input_shape(KernelInput::B, 1);
        int row = index / columns;
        column = index % columns;
        if (operation == IntegerMatrix::PackedLinear) {
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
        const bool transposed = operation == IntegerMatrix::Matmul && p.attribute(0) != 0;
        arow = integer_matrix_batch(batch, KernelInput::A, p) + row * shared;
        bstart = integer_matrix_batch(batch, KernelInput::B, p) + (transposed ? column * shared : column);
        bstride = transposed ? 1 : columns;
    }
    uint64_t value = 0;
    for (int k = 0; k < shared; ++k)
        value += integer_reduction_value(a, arow + k, dtype) * integer_reduction_value(b, bstart + k * bstride, dtype);
    if (operation == IntegerMatrix::Addmm) {
        int addend = 0, flat = index, delta = p.output_rank() - p.input_rank(KernelInput::C);
        for (int d = p.output_rank() - 1; d >= delta; --d) {
            const int coordinate = flat % p.output_shape(d);
            flat /= p.output_shape(d);
            if (p.input_shape(KernelInput::C, d - delta) != 1)
                addend += coordinate * p.input_stride(KernelInput::C, d - delta);
        }
        value = integer_matrix_coefficient(p, 0) * value +
                integer_matrix_coefficient(p, 1) * integer_reduction_value(c, addend, dtype);
    } else if (linear && p.input_rank(KernelInput::C) >= 0) {
        value += integer_reduction_value(c, p.input_rank(KernelInput::C) == 0 ? 0 : column, dtype);
    }
    return value;
}

void validate_integer_matrix_coefficient(float value, DType dtype);
void validate_integer_matrix(IntegerMatrix operation, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_integer_matrix_cpu(IntegerMatrix operation, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_integer_matrix_cuda(IntegerMatrix operation, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
#undef ELMAN_MATRIX_HD
