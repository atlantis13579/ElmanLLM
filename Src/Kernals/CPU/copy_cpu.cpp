#include "kernel.h"

namespace elman {
void launch_copy_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    COPY_INPUT_A(i, i);
  }
}
}  // namespace elman
