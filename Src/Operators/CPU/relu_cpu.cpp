#include "kernel.h"

namespace elman {
void launch_relu_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    if (kernel_integer(p.output_dtype())) {
      const int ix = READ_INT_A(i);
      WRITE_INT_OUTPUT(i, ix > 0 ? ix : 0);
      continue;
    }
    float x = READ_A(i);
    WRITE_OUTPUT(i, EL_MAX(x, 0.0f));
  }
}
}  // namespace elman
