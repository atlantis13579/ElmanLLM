#include "Kernals/float64_normalization_math.h"
#include "Backends/CudaStream.h"
#include <cuda_runtime.h>

namespace elman {
__global__ void float64_normalization_kernel(Float64Normalization op, int rows, double* output, const double* a, const double* b, const double* c, KernelParams p) {
    const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < unsigned(rows)) float64_normalization_row(op, int(row), output, a, b, c, p);
}
void launch_float64_normalization_cuda(Float64Normalization op, uint32_t* output, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const int rows = float64_normalization_rows(op, p);
    if (rows) float64_normalization_kernel<<<(unsigned(rows) + 127u) / 128u, 128, 0, current_cuda_stream()>>>(op, rows,
        reinterpret_cast<double*>(output), reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b), reinterpret_cast<const double*>(c), p);
}
} // namespace elman
