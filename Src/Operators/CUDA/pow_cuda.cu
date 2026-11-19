#include "kernel.cuh"

namespace elman {
__global__ void kernel_pow_cuda(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    if (kernel_integer(p.output_dtype())) {
      const int ix = READ_INT_A(broadcast_index(i, KernelInput::A, p));
      const int iy = READ_INT_B(broadcast_index(i, KernelInput::B, p));
      if (iy < 0) {
        WRITE_INT_OUTPUT(i, ix == 1 ? 1 : ix == -1 ? ((iy & 1) ? -1 : 1) : 0);
        return;
      }
      uint32_t base = uint32_t(ix), value = 1u;
      for (int exponent = iy; exponent > 0; exponent >>= 1) {
        if (exponent & 1) {
          value *= base;
        }
        base *= base;
      }
      WRITE_INT_OUTPUT(i, int(value));
      return;
    }
    float x = READ_A(broadcast_index(i, KernelInput::A, p));
    float y = READ_B(broadcast_index(i, KernelInput::B, p));
    WRITE_OUTPUT(i, EL_POW(x, y));
  }
}
void launch_pow_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_pow_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
