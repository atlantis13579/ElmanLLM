#include "kernel.h"

namespace elman {
void launch_layer_norm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int length = 1;
    for (int axis =
             p.input_rank(KernelInput::A) -
             p.attribute(operator_attr::normalization::kTrailingDimensions);
         axis < p.input_rank(KernelInput::A); ++axis) {
      length *= p.input_shape(KernelInput::A, axis);
    }
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
                float_attribute(p, operator_attr::normalization::kEpsilon));
    int coordinate = i % length;
    if (p.input_rank(KernelInput::B) >= 0) {
      value *= READ_B(coordinate);
    }
    if (p.input_rank(KernelInput::C) >= 0) {
      value += READ_C(coordinate);
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
