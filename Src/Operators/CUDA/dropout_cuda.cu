#include "kernel.cuh"

namespace elman {
__global__ void kernel_dropout_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float value = READ_A(i);
    if (p.attribute(operator_attr::dropout::kTraining) != 0) {
      float probability =
          float_attribute(p, operator_attr::dropout::kProbability);
      value = probability >= 1.0f
                  ? 0.0f
                  : (elman_random(
                         i, p.attribute(operator_attr::dropout::kSeed),
                         p.attribute(operator_attr::dropout::kStreamOffset)) >=
                             probability
                         ? value / (1.0f - probability)
                         : 0.0f);
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_dropout_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_dropout_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
