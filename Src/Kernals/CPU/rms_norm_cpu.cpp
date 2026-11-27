#include "kernel.h"
#include "Core/LocalFunction.h"
#include "row_reduction.h"

namespace elman {
void launch_rms_norm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  if (!p.output_count()) return;
  int length = 1;
  for (int axis = p.output_rank() - p.attribute(kernal_attr::normalization::kTrailingDimensions);
       axis < p.output_rank(); ++axis) length *= p.output_shape(axis);
  for (int base = 0; base < p.output_count(); base += length) {
    LocalFunction square{[&](int j) { float value = READ_A(base + j); return value * value; }};
    float total = 0;
    if (length >= 8) total = contiguous_float_sum(length, square);
    else for (int j = 0; j < length; ++j) total += square(j);
    float inverse = 1.0f / std::sqrt(total / float(length) + float_attribute(p, kernal_attr::normalization::kEpsilon));
    for (int j = 0; j < length; ++j) {
      float value = READ_A(base + j) * inverse;
      if (p.attribute(kernal_attr::normalization::kRounding)) {
        if (p.output_dtype() == int(DType::Float16)) value = numeric::half_to_float(numeric::float_to_half(value));
        if (p.output_dtype() == int(DType::BFloat16)) value = numeric::bfloat_to_float(numeric::float_to_bfloat(value));
      }
      if (p.input_rank(KernelInput::B) >= 0) value *= READ_B(j);
      if (p.input_rank(KernelInput::C) >= 0) value += READ_C(j);
      WRITE_OUTPUT(base + j, value);
    }
  }
}
} // namespace elman
