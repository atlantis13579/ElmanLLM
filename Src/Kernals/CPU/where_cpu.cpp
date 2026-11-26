#include "kernel.h"

namespace elman {
void launch_where_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    const bool select = READ_INT_A(broadcast_index(i, KernelInput::A, p)) != 0;
    const int index =
        broadcast_index(i, select ? KernelInput::B : KernelInput::C, p);
    COPY_TYPED_ELEMENT(out, select ? b : c, i, index, p.output_dtype());
  }
}
}  // namespace elman
