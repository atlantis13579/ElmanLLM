#include "kernel.h"

namespace elman {
void launch_mse_loss_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int count = input_count(KernelInput::A, p);
    float total = 0.0f;
    for (int j = 0; j < count; ++j) {
      float difference = READ_A(j) - READ_B(j);
      total += difference * difference;
    }
    WRITE_OUTPUT(i, total / float(count));
  }
}
}  // namespace elman
