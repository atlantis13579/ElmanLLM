#include "kernel.cuh"

namespace elman {
__global__ void kernel_mse_loss_cuda(uint32_t* out, const uint32_t* a,
                                     const uint32_t* b, const uint32_t* c,
                                     KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int count = input_count(KernelInput::A, p);
    float total = 0.0f;
    for (int j = 0; j < count; ++j) {
      float difference = READ_A(j) - READ_B(j);
      total += difference * difference;
    }
    WRITE_OUTPUT(i, total / float(count));
  }
}
void launch_mse_loss_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_mse_loss_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
