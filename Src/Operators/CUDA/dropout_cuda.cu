#include "kernel.cuh"
#include "Operators/dropout_math.h"
#include "Operators/float64_random_math.h"

namespace elman {
__global__ void kernel_dropout_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    const double probability = float64_random_parameter(p, 0);
    if (!p.attribute(operator_attr::dropout::kTraining) || probability == 0.) {
      COPY_TYPED_ELEMENT(out, a, i, i, p.output_dtype());
      return;
    }
    const bool keep = probability < 1. && double(elman_random(i, p.attribute(0), p.attribute(1))) >= probability;
    const float scale = floating_dropout_scale(DType(p.output_dtype()), probability, p.attribute(operator_attr::dropout::kBackward) != 0);
    WRITE_OUTPUT(i, READ_A(i) * (keep ? scale : 0.f));
  }
}
void launch_dropout_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_dropout_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
