#include "kernel.cuh"

#include "Kernals/gelu_params.h"

namespace elman {
__global__ void kernel_gelu_cuda(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c,
                                 KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float x = READ_A(i);
    if (p.attribute(kernal_attr::gelu::kExact)) {
      WRITE_OUTPUT(i, 0.5f * x * (1.0f + erff(x * 0.7071067811865475f)));
      return;
    }
    WRITE_OUTPUT(i, 0.5f * x *
                        (1.0f + EL_TANH(0.7978845608028654f *
                                        (x + 0.044715f * x * x * x))));
  }
}
void launch_gelu_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_gelu_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
