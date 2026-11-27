#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Kernals/group_norm_params.h"

namespace elman {
void launch_group_norm_input_grad_cpu(uint32_t* out, const uint32_t* a,
                                      const uint32_t* b, const uint32_t* c,
                                      const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
    int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
    int per_group = channels / p.attribute(kernal_attr::group_norm::kGroups);
    int length = per_group * spatial;
    int base = (i / length) * length;
    float average = 0;
    for (int j = 0; j < length; ++j) average += READ_A(base + j);
    average /= float(length);
    float variance = 0;
    for (int j = 0; j < length; ++j) {
      float difference = READ_A(base + j) - average;
      variance += difference * difference;
    }
    variance /= float(length);
    float epsilon = float_attribute(p, kernal_attr::group_norm::kEpsilon);
    float inverse = p.input_dtype(KernelInput::A) == int(DType::Float32) ?
        float(1.0 / std::sqrt(double(variance) + group_norm_epsilon(p))) : 1.0f / EL_SQRT(variance + epsilon);
    const DType dtype = static_cast<DType>(p.input_dtype(KernelInput::B));
    if (is_narrow_float(dtype)) {
      LocalFunction round{[dtype](float v) {
        return dtype == DType::Float16 ? numeric::half_to_float(numeric::float_to_half(v)) :
                                        numeric::bfloat_to_float(numeric::float_to_bfloat(v));
      }};
      average = round(average);
      inverse = round(inverse);
    }
    float ds = 0, db = 0;
    for (int channel = 0; channel < per_group; ++channel) {
      float channel_ds = 0, channel_db = 0;
      for (int j = 0; j < spatial; ++j) {
        int index = base + channel * spatial + j;
        channel_ds += READ_C(index) * READ_A(index);
        channel_db += READ_C(index);
      }
      float weight = READ_B((base / spatial + channel) % channels);
      ds += channel_ds * weight;
      db += channel_db * weight;
    }
    float coefficient2 = (db * average - ds) * inverse * inverse * inverse / float(length);
    float coefficient3 = -coefficient2 * average - db * inverse / float(length);
    float coefficient1 = inverse * READ_B((i / spatial) % channels);
    WRITE_OUTPUT(i, coefficient1 * READ_C(i) + coefficient2 * READ_A(i) + coefficient3);
  }
}
} // namespace elman
