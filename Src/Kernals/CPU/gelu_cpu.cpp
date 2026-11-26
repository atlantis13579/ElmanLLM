#include "kernel.h"

#include "Kernals/gelu_params.h"

namespace elman {
void launch_gelu_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float x = READ_A(i);
    if (p.attribute(operator_attr::gelu::kExact)) {
      WRITE_OUTPUT(i, 0.5f * x * (1.0f + std::erf(x * 0.7071067811865475f)));
      continue;
    }
    WRITE_OUTPUT(i, 0.5f * x *
                        (1.0f + EL_TANH(0.7978845608028654f *
                                        (x + 0.044715f * x * x * x))));
  }
}
}  // namespace elman
