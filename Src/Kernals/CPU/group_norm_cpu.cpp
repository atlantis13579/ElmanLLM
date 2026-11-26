#include "kernel.h"
#include "Kernals/group_norm_params.h"

namespace elman {
void launch_group_norm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
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
    if (p.input_dtype(KernelInput::A) == int(DType::Float32)) {
      float inverse = float(1.0 / std::sqrt(double(variance) + group_norm_epsilon(p)));
      int channel = (i / spatial) % channels;
      float scale = inverse * (p.input_rank(KernelInput::B) >= 0 ? READ_B(channel) : 1.0f);
      float bias = (p.input_rank(KernelInput::C) >= 0 ? READ_C(channel) : 0.0f) - scale * average;
      WRITE_OUTPUT(i, scale * READ_A(i) + bias);
      continue;
    }
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
}  // namespace elman
