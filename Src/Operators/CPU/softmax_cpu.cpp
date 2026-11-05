#include "kernel.h"

namespace elman {
void launch_softmax_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int axis = p.attribute(operator_attr::softmax::kAxis);
    int inner = p.input_stride(KernelInput::A, axis);
    int length = p.input_shape(KernelInput::A, axis);
    int base = (i / (length * inner)) * length * inner + i % inner;
    float maximum_value = -EL_INF;
    for (int j = 0; j < length; ++j) {
      maximum_value = EL_MAX(maximum_value, READ_A(base + j * inner));
    }
    float denominator = 0.0f;
    for (int j = 0; j < length; ++j) {
      denominator += EL_EXP(READ_A(base + j * inner) - maximum_value);
    }
    WRITE_OUTPUT(i, EL_EXP(READ_A(i) - maximum_value) / denominator);
  }
}
}  // namespace elman
