#include "kernel.cuh"

namespace elman {
__global__ void kernel_sub_cuda(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    if (kernel_integer(p.output_dtype())) {
      const int ix = READ_INT_A(broadcast_index(i, KernelInput::A, p));
      const int iy = READ_INT_B(broadcast_index(i, KernelInput::B, p));
      WRITE_INT_OUTPUT(i, int(uint32_t(ix) - uint32_t(iy)));
      return;
    }
    float x = READ_A(broadcast_index(i, KernelInput::A, p));
    float y = READ_B(broadcast_index(i, KernelInput::B, p));
    WRITE_OUTPUT(i, x - y);
  }
}
void launch_sub_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_sub_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
