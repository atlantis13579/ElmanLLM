#include "kernel.h"

namespace elman {
void launch_normal_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float u = elman_random(2 * i, p.attribute(operator_attr::random::kSeed),
                           p.attribute(operator_attr::random::kStreamOffset));
    float v = elman_random(2 * i + 1, p.attribute(operator_attr::random::kSeed),
                           p.attribute(operator_attr::random::kStreamOffset));
    WRITE_OUTPUT(
        i, float_attribute(p, operator_attr::random::kMean) +
               float_attribute(p, operator_attr::random::kStandardDeviation) *
                   EL_SQRT(-2.0f * EL_LOG(u)) * EL_COS(6.283185307179586f * v));
  }
}
}  // namespace elman
