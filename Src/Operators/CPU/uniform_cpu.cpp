#include "kernel.h"

namespace elman {
void launch_uniform_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float random_value =
        elman_random(i, p.attribute(operator_attr::random::kSeed),
                     p.attribute(operator_attr::random::kStreamOffset));
    WRITE_OUTPUT(i,
                 float_attribute(p, operator_attr::random::kLowerBound) +
                     (float_attribute(p, operator_attr::random::kUpperBound) -
                      float_attribute(p, operator_attr::random::kLowerBound)) *
                         random_value);
  }
}
}  // namespace elman
