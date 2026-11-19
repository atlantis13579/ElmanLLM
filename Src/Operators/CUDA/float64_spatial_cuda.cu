#include "kernel.cuh"
#include "Operators/float64_spatial_math.h"

namespace elman {
namespace {
__global__ void float64_spatial_kernel(Float64Spatial op, double* out, const double* a,
                                      const double* b, const double* c, KernelParams p) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < p.output_count()) out[i] = float64_spatial_value(op, i, a, b, c, p);
}
}
void launch_float64_spatial_cuda(Float64Spatial op, uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (p.output_count()) float64_spatial_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>
        (op, reinterpret_cast<double*>(out), reinterpret_cast<const double*>(a), reinterpret_cast<const double*>(b),
         reinterpret_cast<const double*>(c), p);
}
} // namespace elman
