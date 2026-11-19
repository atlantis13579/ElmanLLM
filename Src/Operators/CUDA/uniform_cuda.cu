#include "kernel.cuh"
#include "Operators/floating_random_math.h"

namespace elman {
__global__ void kernel_uniform_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    WRITE_OUTPUT(i, floating_random_value(false, i, p));
  }
}
void launch_uniform_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_uniform_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
