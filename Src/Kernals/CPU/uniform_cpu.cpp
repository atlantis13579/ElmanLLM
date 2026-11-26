#include "kernel.h"
#include "Kernals/floating_random_math.h"

namespace elman {
void launch_uniform_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    WRITE_OUTPUT(i, floating_random_value(false, i, p));
  }
}
}  // namespace elman
