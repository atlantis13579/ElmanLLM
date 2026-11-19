#include "kernel.h"

namespace elman {
void launch_abs_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    if (kernel_integer(p.output_dtype())) {
      const int ix = READ_INT_A(i);
      WRITE_INT_OUTPUT(i, ix < 0 ? int(0u - uint32_t(ix)) : ix);
      continue;
    }
    float x = READ_A(i);
    WRITE_OUTPUT(i, EL_ABS(x));
  }
}
}  // namespace elman
