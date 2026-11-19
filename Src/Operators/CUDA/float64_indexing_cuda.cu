#include "kernel.cuh"
#include "Operators/float64_indexing_math.h"

namespace elman {
namespace {
__global__ void float64_indexing_kernel(Float64Indexing op, double* out, const double* a,
                                       const int32_t* ids, const double* c, KernelParams p) {
    const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < p.output_count()) {
        // Validated permutations need no atomics and preserve the input bits.
        if (op == Float64Indexing::SortGradient) out[float64_sort_target(i, ids, p)] = a[i];
        else out[i] = float64_indexing_value(op, i, a, ids, c, p);
    }
}
}
void launch_float64_indexing_cuda(Float64Indexing op, uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    if (p.output_count()) float64_indexing_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>
        (op, reinterpret_cast<double*>(out), reinterpret_cast<const double*>(a), reinterpret_cast<const int32_t*>(b),
         reinterpret_cast<const double*>(c), p);
}
} // namespace elman
