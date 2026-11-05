#include "kernel.h"

namespace elman {
void launch_cross_entropy_grad_cpu(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int rows = p.input_shape(KernelInput::A, tensor_axis::logits::kBatch);
    int columns = p.input_shape(KernelInput::A, tensor_axis::logits::kClasses);
    int row = i / columns;
    int column = i % columns;
    float maximum_value = -EL_INF;
    for (int j = 0; j < columns; ++j) {
      maximum_value = EL_MAX(maximum_value, READ_A(row * columns + j));
    }
    float denominator = 0.0f;
    for (int j = 0; j < columns; ++j) {
      denominator += EL_EXP(READ_A(row * columns + j) - maximum_value);
    }
    float value = EL_EXP(READ_A(i) - maximum_value) / denominator -
                  float(column == READ_INT_B(row));
    WRITE_OUTPUT(i, value * READ_C(0) / float(rows));
  }
}
}  // namespace elman
