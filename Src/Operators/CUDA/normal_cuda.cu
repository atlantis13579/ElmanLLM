#include "kernel.cuh"

namespace elman {
__global__ void kernel_normal_cuda(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float u = elman_random(2 * i, p.attribute(operator_attr::random::kSeed),
                           p.attribute(operator_attr::random::kStreamOffset));
    float v = elman_random(2 * i + 1, p.attribute(operator_attr::random::kSeed),
                           p.attribute(operator_attr::random::kStreamOffset));
    WRITE_OUTPUT(
        i, float_attribute(p, operator_attr::random::kMean) +
               float_attribute(p, operator_attr::random::kStandardDeviation) *
                   EL_SQRT(-2.0f * EL_LOG(u)) * EL_COS(6.283185307179586f * v));
  }
}
void launch_normal_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_normal_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
