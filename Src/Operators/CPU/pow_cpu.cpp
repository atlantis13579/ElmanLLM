#include "kernel.h"

namespace elman {
void launch_pow_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    if (kernel_integer(p.output_dtype())) {
      const int ix = READ_INT_A(broadcast_index(i, KernelInput::A, p));
      const int iy = READ_INT_B(broadcast_index(i, KernelInput::B, p));
      if (iy < 0) {
        WRITE_INT_OUTPUT(i, ix == 1 ? 1 : ix == -1 ? ((iy & 1) ? -1 : 1) : 0);
        continue;
      }
      uint32_t base = uint32_t(ix), value = 1u;
      for (int exponent = iy; exponent > 0; exponent >>= 1) {
        if (exponent & 1) {
          value *= base;
        }
        base *= base;
      }
      WRITE_INT_OUTPUT(i, int(value));
      continue;
    }
    float x = READ_A(broadcast_index(i, KernelInput::A, p));
    float y = READ_B(broadcast_index(i, KernelInput::B, p));
    WRITE_OUTPUT(i, EL_POW(x, y));
  }
}
}  // namespace elman
