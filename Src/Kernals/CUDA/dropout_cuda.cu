#include "kernel.cuh"
#include "Kernals/dropout_math.h"
#include "Kernals/float64_random_math.h"
#include <limits>

namespace elman {
__global__ void float_dropout_kernel(float* output, const float* input, int count, int seed, int offset,
                                     float threshold, float scale, bool training) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) output[i] = training ? input[i] * (elman_random(i, seed, offset) >= threshold ? scale : 0.f) : input[i];
}
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
    if (p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == int(DType::Float32)) {
      const double probability = float64_random_parameter(p, 0);
      float threshold = float(probability);
      // Compare a Float32 random sample with the original Double probability exactly.
      if (double(threshold) < probability) threshold = std::nextafter(threshold, 1.f);
      if (probability == 1.) threshold = std::numeric_limits<float>::infinity();
      const float scale = probability == 1. ? 0.f : p.attribute(operator_attr::dropout::kBackward) ?
          float(1. / (1. - probability)) : 1.f / float(1. - probability);
      float_dropout_kernel<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
          reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a), p.output_count(), p.attribute(0), p.attribute(1),
          threshold, scale, p.attribute(operator_attr::dropout::kTraining) && probability != 0.);
      return;
    }
    kernel_dropout_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
