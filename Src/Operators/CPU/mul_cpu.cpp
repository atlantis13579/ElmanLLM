#include "Backends/CpuParallel.h"
#include "kernel.h"

namespace elman {
void launch_mul_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  parallel_for_cpu(p.output_count(), 8192, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      if (kernel_integer(p.output_dtype())) {
        const int ix = READ_INT_A(broadcast_index(i, KernelInput::A, p));
        const int iy = READ_INT_B(broadcast_index(i, KernelInput::B, p));
        WRITE_INT_OUTPUT(i, int(uint32_t(ix) * uint32_t(iy)));
        continue;
      }
      float x = READ_A(broadcast_index(i, KernelInput::A, p));
      float y = READ_B(broadcast_index(i, KernelInput::B, p));
      WRITE_OUTPUT(i, x * y);
    }
  });
}
}  // namespace elman
