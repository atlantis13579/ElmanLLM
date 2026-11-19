#pragma once
#include "Operators/integer_extrema.h"

namespace elman {
template<bool Maximum>
__global__ void integer_extrema_cuda(uint32_t* output, const uint32_t* input, KernelParams p) {
    const int row = int(blockIdx.x), lane = int(threadIdx.x);
    const int axis = p.attribute(operator_attr::reduction::kAxis);
    const int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    const int length = axis < 0 ? input_count(KernelInput::A, p) : p.input_shape(KernelInput::A, axis);
    const int base = axis < 0 ? 0 : (row / inner) * length * inner + row % inner;
    const int dtype = p.input_dtype(KernelInput::A);
    // Validation rejects empty reduction axes, including empty-output shapes.
    uint64_t value = integer_reduction_value(input, base, dtype);
    for (int64_t j = lane; j < length; j += 256)
        value = integer_extremum<Maximum>(value,
            integer_reduction_value(input, int(int64_t(base) + j * inner), dtype), dtype);
    __shared__ uint64_t values[256];
    values[lane] = value;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (lane < offset) values[lane] = integer_extremum<Maximum>(values[lane], values[lane + offset], dtype);
        __syncthreads();
    }
    if (!lane) integer_write(output, row, dtype, values[0]);
}
} // namespace elman
