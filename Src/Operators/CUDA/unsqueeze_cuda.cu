#include "kernel.cuh"

namespace elman {
__global__ void kernel_unsqueeze_cuda(uint32_t* out, const uint32_t* a,
                                      const uint32_t* b, const uint32_t* c,
                                      KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    COPY_INPUT_A(i, i);
  }
}
void launch_unsqueeze_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_unsqueeze_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
