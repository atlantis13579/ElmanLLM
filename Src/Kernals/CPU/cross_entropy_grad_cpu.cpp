#include "kernel.h"
#include "Kernals/low_precision_loss.h"

namespace elman {
void launch_cross_entropy_grad_cpu(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   const KernelParams& p) {
  if (p.input_dtype(KernelInput::A) == int(DType::Float32) && p.output_dtype() == int(DType::Float32)) {
    launch_low_precision_loss_cpu(LowPrecisionLoss::CrossEntropyGradient, out, a, b, c, p);
    return;
  }
  const int rows = p.input_shape(KernelInput::A, tensor_axis::logits::kBatch);
  const int columns =
      p.input_shape(KernelInput::A, tensor_axis::logits::kClasses);
  for (int row = 0; row < rows; ++row) {
    float maximum_value = -EL_INF;
    for (int j = 0; j < columns; ++j) {
      maximum_value = EL_MAX(maximum_value, READ_A(row * columns + j));
    }
    float denominator = 0.0f;
    for (int j = 0; j < columns; ++j) {
      denominator += EL_EXP(READ_A(row * columns + j) - maximum_value);
    }
    for (int column = 0; column < columns; ++column) {
      const int i = row * columns + column;
      float value = EL_EXP(READ_A(i) - maximum_value) / denominator -
                    float(column == READ_INT_B(row));
      WRITE_OUTPUT(i, value * READ_C(0) / float(rows));
    }
  }
}
}  // namespace elman
