#include "kernel.h"

namespace elman {
void launch_exp_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float x = READ_A(i);
    WRITE_OUTPUT(i, EL_EXP(x));
  }
}
}  // namespace elman
