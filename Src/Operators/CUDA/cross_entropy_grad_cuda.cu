#include "../../Backends/cuda_kernel.cuh"
#include "../../Backends/cuda_row_reduce.cuh"

namespace elman {
__global__ void cross_entropy_gradient_rows(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                          const uint32_t* c, KernelParams p) {
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
    int target = IB(row);
    float scale = C(0) / float(P(11));
    for (int column = int(threadIdx.x); column < columns; column += int(blockDim.x)) {
        int i = row * columns + column;
        W(i, (expf(A(i) - highest) / denominator - float(column == target)) * scale);
    }
}
void launch_cross_entropy_grad_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                   const uint32_t* c, const KernelParams& p) {
    cross_entropy_gradient_rows<<<P(11), 256>>>(out, a, b, c, p);
}
}
