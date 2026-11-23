#include "kernel.cuh"
#include "Operators/low_precision_attention.h"
#include "Operators/low_attention_reduction.h"
#include <algorithm>
namespace elman {
namespace {
__device__ float low_attention_sum_reduce(float value, float* shared, int width) {
    const int thread = int(threadIdx.x), lane = thread % width, row = thread / width, height = int(blockDim.x) / width;
    shared[thread] = value; __syncthreads();
    for (int offset = height / 2; offset; offset /= 2) {
        if (row < offset) shared[thread] += shared[thread + offset * width];
        __syncthreads();
    }
    for (int offset = width / 2; offset >= 32; offset /= 2) {
        if (!row && lane < offset) shared[thread] += shared[thread + offset];
        __syncthreads();
    }
    value = shared[thread];
    const int warp_width = width < 32 ? width : 32;
    for (int offset = 1; offset < warp_width; offset *= 2) value += __shfl_down_sync(__activemask(), value, offset, warp_width);
    return value;
}
__global__ void low_attention_sum_kernel(float* output, const float* input, AttentionReduction r, int width, bool vectorized) {
    __shared__ float partial[512];
    const int row = int(blockIdx.x), thread = int(threadIdx.x), step = int(blockDim.x);
    float values[4]{};
    const int length = r.length, base = r.offset(row, 0);
    const int shift = vectorized ? base % 4 : 0, head = shift ? 4 - shift : 0;
    if (vectorized) {
        if (shift && thread < 4 && thread >= shift) values[0] += input[base + thread - shift];
        for (int j = thread * 4 + head; j + 3 < length; j += step * 4)
            for (int k = 0; k < 4; ++k) values[k] += input[base + j + k];
        const int tail = head + (length - head) / 4 * 4 + thread;
        if (thread < width && tail < length) values[0] += input[base + tail];
    } else for (int j = thread; j < length; j += step * 4)
        for (int k = 0; k < 4 && j + k * step < length; ++k) values[k] += input[r.offset(row, j + k * step)];
    const float sum = low_attention_sum_reduce(((values[0] + values[1]) + values[2]) + values[3], partial, width);
    if (!thread) output[row] = sum;
}
}
void launch_low_attention_sum_cuda(uint32_t* output, const uint32_t* input, const KernelParams& p) {
    const AttentionReduction r(p);
    const int length = r.length, rows = p.output_count();
    if (!rows) return;
    if (!length) {
        check_cuda_execution(cudaMemsetAsync(output, 0, size_t(rows) * sizeof(float), current_cuda_stream()), "attention empty reduction");
        return;
    }
    int (*const power)(int) = [](int value) { int n = 1; while (n <= value / 2 && n < 512) n *= 2; return n; };
    const bool vectorized = r.fastest && r.reduced_blocks == 1 && length > 128;
    const int output_vector = r.fastest ? 1 : r.inner % 4 == 0 ? 4 : r.inner % 2 == 0 ? 2 : 1;
    const int limit = 512 / output_vector;
    const int first = std::min(limit, power(r.fastest ? length / (vectorized ? 4 : 1) : rows / output_vector));
    const int second = std::min(limit, power(r.fastest ? rows : length));
    int width = std::min(first, 32), height = std::min(second, limit / width);
    width = std::min(first, limit / height);
    const int step = r.fastest ? width : 1;
    if ((length + step - 1) / step < std::min(height * 16, 256)) height = 1;
    if (!r.fastest) width = 1;
    low_attention_sum_kernel<<<rows, width * height, 0, current_cuda_stream()>>>(reinterpret_cast<float*>(output), reinterpret_cast<const float*>(input), r, width, vectorized);
}
} // namespace elman
