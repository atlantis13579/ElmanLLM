#include "Kernals/float64_random_math.h"
#include "Backends/CudaStream.h"
#include <cuda_runtime.h>

namespace elman {
__global__ void float64_random_kernel(Float64Random op, double* output, const double* input, KernelParams p) {
    const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < unsigned(p.output_count())) output[i] = float64_random_value(op, uint64_t(i), op == Float64Random::Dropout ? input[i] : 0., p);
}
void launch_float64_random_cuda(Float64Random op, uint32_t* output, const uint32_t* input, const KernelParams& p) {
    if (p.output_count()) float64_random_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        op, reinterpret_cast<double*>(output), reinterpret_cast<const double*>(input), p);
}
} // namespace elman
