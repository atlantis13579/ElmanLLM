#include "kernel.h"

namespace elman {
void launch_reduce_grad_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int axis = p.attribute(operator_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length =
        axis < 0 ? p.output_count() : p.input_shape(KernelInput::A, axis);
    int target = axis < 0 ? 0 : (i / (length * inner)) * inner + i % inner;
    float value = READ_C(target);
    int kind = p.attribute(operator_attr::reduction::kGradientKind);
    if (kind == 1) {
      value /= float(length);
    }
    if (kind == 2 || kind == 3) {
      float extremum = READ_B(target);
      int base =
          axis < 0 ? 0 : (target / inner) * length * inner + target % inner;
      int matches = 0;
      for (int j = 0; j < length; ++j) {
        if (READ_A(base + j * inner) == extremum) {
          ++matches;
        }
      }
      value = READ_A(i) == extremum ? value / float(matches) : 0.0f;
    }
    if (kind == 4) {
      int base =
          axis < 0 ? 0 : (target / inner) * length * inner + target % inner;
      float product = 1.0f;
      for (int j = 0; j < length; ++j) {
        if (base + j * inner != i) {
          product *= READ_A(base + j * inner);
        }
      }
      value *= product;
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
