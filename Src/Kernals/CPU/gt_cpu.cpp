#include "kernel.h"

namespace elman {
void launch_gt_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                   const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    if (kernel_integer(p.input_dtype(KernelInput::A)) &&
        kernel_integer(p.input_dtype(KernelInput::B))) {
      const int ix = READ_INT_A(broadcast_index(i, KernelInput::A, p));
      const int iy = READ_INT_B(broadcast_index(i, KernelInput::B, p));
      WRITE_INT_OUTPUT(i, ix > iy);
      continue;
    }
    float x = READ_A(broadcast_index(i, KernelInput::A, p));
    float y = READ_B(broadcast_index(i, KernelInput::B, p));
    WRITE_OUTPUT(i, float(x > y));
  }
}
}  // namespace elman
