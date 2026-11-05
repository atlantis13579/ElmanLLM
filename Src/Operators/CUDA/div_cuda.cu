#include "kernel.cuh"

namespace elman {
__global__ void kernel_div_cuda(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float x = READ_A(broadcast_index(i, KernelInput::A, p));
    float y = READ_B(broadcast_index(i, KernelInput::B, p));
    WRITE_OUTPUT(i, x / y);
  }
}
void launch_div_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_div_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
