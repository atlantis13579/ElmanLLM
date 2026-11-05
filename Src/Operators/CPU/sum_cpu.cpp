#include "kernel.h"

namespace elman {
void launch_sum_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int axis = p.attribute(operator_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = axis < 0 ? input_count(KernelInput::A, p)
                          : p.input_shape(KernelInput::A, axis);
    int base = axis < 0 ? 0 : (i / inner) * length * inner + i % inner;
    float value = 0.0f;
    for (int j = 0; j < length; ++j) {
      float x = READ_A(base + j * inner);
      value = value + x;
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
