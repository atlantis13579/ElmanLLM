#include "kernel.h"

namespace elman {
void launch_cast_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    if (p.input_dtype(KernelInput::A) == p.output_dtype()) {
      COPY_INPUT_A(i, i);
      continue;
    }
    if (kernel_integer(p.input_dtype(KernelInput::A))) {
      WRITE_INT_OUTPUT(i, READ_INT_A(i));
      continue;
    }
    float x = READ_A(i);
    WRITE_OUTPUT(i, x);
  }
}
}  // namespace elman
