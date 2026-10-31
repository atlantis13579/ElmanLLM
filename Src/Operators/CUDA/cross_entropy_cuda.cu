#include "../../Backends/cuda_kernel.cuh"
#include "../../Backends/cuda_row_reduce.cuh"
#include <stdexcept>

namespace elman {
__global__ void cross_entropy_rows(float* losses, const uint32_t* a, const uint32_t* b, KernelParams p) {
    __shared__ float scratch[256];
    int row = int(blockIdx.x), columns = P(12);
    float highest = -EL_INF;
    for (int column = int(threadIdx.x); column < columns; column += int(blockDim.x))
        highest = fmaxf(highest, A(row * columns + column));
    highest = row_reduce<true>(highest, scratch);
    float denominator = 0;
    for (int column = int(threadIdx.x); column < columns; column += int(blockDim.x))
        denominator += expf(A(row * columns + column) - highest);
    denominator = row_reduce<false>(denominator, scratch);
    if (threadIdx.x == 0) {
        int target = IB(row);
        losses[row] = target >= 0 && target < columns ? logf(denominator) + highest - A(row * columns + target)
                                                     : EL_INF;
    }
}
__global__ void cross_entropy_average(uint32_t* out, const float* losses, KernelParams p) {
    __shared__ float scratch[256];
    float value = 0;
    for (int row = int(threadIdx.x); row < P(11); row += int(blockDim.x)) value += losses[row];
    value = row_reduce<false>(value, scratch);
    if (threadIdx.x == 0) W(0, value / float(P(11)));
}
void launch_cross_entropy_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t*, const KernelParams& p) {
    float* losses = nullptr;
    auto status = cudaMalloc(reinterpret_cast<void**>(&losses), size_t(P(11)) * sizeof(float));
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    cross_entropy_rows<<<P(11), 256>>>(losses, a, b, p);
    cross_entropy_average<<<1, 256>>>(out, losses, p);
    status = cudaGetLastError();
    auto released = cudaFree(losses);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    if (released != cudaSuccess) throw std::runtime_error(cudaGetErrorString(released));
}
}
