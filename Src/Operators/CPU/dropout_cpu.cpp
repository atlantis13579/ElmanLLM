#include "kernel.h"
#include "Operators/dropout_math.h"
#include "Operators/float64_random_math.h"

namespace elman {
void launch_dropout_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    const double probability = float64_random_parameter(p, 0);
    if (!p.attribute(operator_attr::dropout::kTraining) || probability == 0.) {
      COPY_TYPED_ELEMENT(out, a, i, i, p.output_dtype());
      continue;
    }
    const bool keep = probability < 1. && double(elman_random(i, p.attribute(0), p.attribute(1))) >= probability;
    const float scale = floating_dropout_scale(DType(p.output_dtype()), probability, p.attribute(operator_attr::dropout::kBackward) != 0);
    WRITE_OUTPUT(i, READ_A(i) * (keep ? scale : 0.f));
  }
}
}  // namespace elman
