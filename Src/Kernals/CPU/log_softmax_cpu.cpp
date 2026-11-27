#include "kernel.h"

namespace elman {
void launch_log_softmax_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  const int axis = p.attribute(kernal_attr::softmax::kAxis);
  const int inner = p.input_rank(KernelInput::A) == 0
                        ? 1
                        : p.input_stride(KernelInput::A, axis);
  const int length = p.input_rank(KernelInput::A) == 0
                         ? 1
                         : p.input_shape(KernelInput::A, axis);
  if (p.output_count() == 0) {
    return;
  }
  for (int group = 0; group < p.output_count() / length; ++group) {
    const int base = (group / inner) * length * inner + group % inner;
    float maximum_value = -EL_INF;
    for (int j = 0; j < length; ++j) {
      maximum_value = EL_MAX(maximum_value, READ_A(base + j * inner));
    }
    float denominator = 0.0f;
    for (int j = 0; j < length; ++j) {
      denominator += EL_EXP(READ_A(base + j * inner) - maximum_value);
    }
    for (int j = 0; j < length; ++j) {
      const int i = base + j * inner;
      WRITE_OUTPUT(i, READ_A(i) - maximum_value - EL_LOG(denominator));
    }
  }
}
}  // namespace elman
