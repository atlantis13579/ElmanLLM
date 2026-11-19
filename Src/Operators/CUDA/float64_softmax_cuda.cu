#include "Operators/float64_softmax_math.h"
#include "Backends/CudaStream.h"
#include "row_reduce.cuh"
#include <cuda_runtime.h>

namespace elman {
namespace {
constexpr int row_threads = 128;
__global__ void float64_softmax_contiguous(Float64Softmax op, double* output,
                                           const double* input, int length) {
    __shared__ double scratch[row_threads];
    const int base = int(blockIdx.x) * length;
    double maximum = -INFINITY;
    for (int j = threadIdx.x; j < length; j += row_threads) maximum = fmax(maximum, input[base + j]);
    maximum = row_reduce<true>(maximum, scratch);
    double sum = 0;
    for (int j = threadIdx.x; j < length; j += row_threads) sum += exp(input[base + j] - maximum);
    sum = row_reduce<false>(sum, scratch);
    const bool logarithmic = op == Float64Softmax::LogSoftmax;
    const double logarithm = logarithmic ? log(sum) : 0;
    for (int j = threadIdx.x; j < length; j += row_threads) {
        const double shifted = input[base + j] - maximum;
        output[base + j] = logarithmic ? shifted - logarithm : exp(shifted) / sum;
    }
}
} // namespace
__global__ void float64_softmax_kernel(Float64Softmax op, double* output, const double* input, int rows, int length, int inner) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < unsigned(rows)) float64_softmax_row(op, output, input, int(row), length, inner);
}
void launch_float64_softmax_cuda(Float64Softmax op, uint32_t* output, const uint32_t* input, const KernelParams& p) {
    if (!p.output_count()) return;
    const int axis = p.attribute(0), rank = p.input_rank(KernelInput::A);
    const int length = rank ? p.input_shape(KernelInput::A, axis) : 1;
    const int inner = rank ? p.input_stride(KernelInput::A, axis) : 1;
    const int rows = p.output_count() / length;
    if (inner == 1 && length >= 128) {
        float64_softmax_contiguous<<<rows, row_threads, 0, current_cuda_stream()>>>(
            op, reinterpret_cast<double*>(output), reinterpret_cast<const double*>(input), length);
        return;
    }
    float64_softmax_kernel<<<(unsigned(rows) + 127u) / 128u, 128, 0, current_cuda_stream()>>>(
        op, reinterpret_cast<double*>(output), reinterpret_cast<const double*>(input), rows, length, inner);
}
} // namespace elman
