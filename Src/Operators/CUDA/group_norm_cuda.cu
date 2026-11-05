#include "kernel.cuh"

namespace elman {
__global__ void kernel_group_norm_cuda(uint32_t* out, const uint32_t* a,
                                       const uint32_t* b, const uint32_t* c,
                                       KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
    int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
    int per_group = channels / p.attribute(operator_attr::group_norm::kGroups);
    int length = per_group * spatial;
    int base = (i / length) * length;
    float average = 0.0f;
    for (int j = 0; j < length; ++j) {
      average += READ_A(base + j);
    }
    average /= float(length);
    float variance = 0.0f;
    for (int j = 0; j < length; ++j) {
      float difference = READ_A(base + j) - average;
      variance += difference * difference;
    }
    variance /= float(length);
    float value =
        (READ_A(i) - average) /
        EL_SQRT(variance +
                float_attribute(p, operator_attr::group_norm::kEpsilon));
    int channel = (i / spatial) % channels;
    if (p.input_rank(KernelInput::B) >= 0) {
      value *= READ_B(channel);
    }
    if (p.input_rank(KernelInput::C) >= 0) {
      value += READ_C(channel);
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_group_norm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_group_norm_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
