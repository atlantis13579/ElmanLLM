#pragma once
#include "Kernals/integer_reduction.h"

namespace elman {
template <bool Product>
__global__ void integer_reduction_cuda(uint64_t* output, const uint32_t* input, KernelParams p) {
    int row = int(blockIdx.x), lane = int(threadIdx.x);
    int axis = p.attribute(kernal_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = axis < 0 ? input_count(KernelInput::A, p) : p.input_shape(KernelInput::A, axis);
    int base = axis < 0 ? 0 : (row / inner) * length * inner + row % inner;
    uint64_t value = Product ? 1 : 0;
    for (int64_t j = lane; j < length; j += 256) {
        uint64_t item = integer_reduction_value(input, int(int64_t(base) + j * inner), p.input_dtype(KernelInput::A));
        value = Product ? value * item : value + item;
    }
    __shared__ uint64_t values[256];
    values[lane] = value;
    __syncthreads();
    for (int offset = 128; offset; offset /= 2) {
        if (lane < offset)
            values[lane] = Product ? values[lane] * values[lane + offset] : values[lane] + values[lane + offset];
        __syncthreads();
    }
    if (!lane) output[row] = values[0];
}
} // namespace elman
