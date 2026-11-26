#include "kernel.cuh"

namespace elman {
__global__ void kernel_abs_cuda(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    if (kernel_integer(p.output_dtype())) {
      const int ix = READ_INT_A(i);
      WRITE_INT_OUTPUT(i, ix < 0 ? int(0u - uint32_t(ix)) : ix);
      return;
    }
    float x = READ_A(i);
    WRITE_OUTPUT(i, EL_ABS(x));
  }
}
void launch_abs_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_abs_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
