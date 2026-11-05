#include "kernel.h"

namespace elman {
void launch_dropout_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float value = READ_A(i);
    if (p.attribute(operator_attr::dropout::kTraining) != 0) {
      float probability =
          float_attribute(p, operator_attr::dropout::kProbability);
      value = probability >= 1.0f
                  ? 0.0f
                  : (elman_random(
                         i, p.attribute(operator_attr::dropout::kSeed),
                         p.attribute(operator_attr::dropout::kStreamOffset)) >=
                             probability
                         ? value / (1.0f - probability)
                         : 0.0f);
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
