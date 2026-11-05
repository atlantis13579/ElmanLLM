#include "kernel.cuh"

namespace elman {
__global__ void kernel_uniform_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float random_value =
        elman_random(i, p.attribute(operator_attr::random::kSeed),
                     p.attribute(operator_attr::random::kStreamOffset));
    WRITE_OUTPUT(i,
                 float_attribute(p, operator_attr::random::kLowerBound) +
                     (float_attribute(p, operator_attr::random::kUpperBound) -
                      float_attribute(p, operator_attr::random::kLowerBound)) *
                         random_value);
  }
}
void launch_uniform_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_uniform_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
