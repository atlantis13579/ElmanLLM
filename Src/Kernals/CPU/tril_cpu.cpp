#include "kernel.h"

namespace elman {
void launch_tril_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int columns = p.output_shape(p.output_rank() - 1);
    int rows = p.output_shape(p.output_rank() - 2);
    int row = (i / columns) % rows;
    int column = i % columns;
    if (column - row <= p.attribute(kernal_attr::triangular::kDiagonal)) {
      COPY_INPUT_A(i, i);
    } else {
      WRITE_OUTPUT(i, 0.0f);
    }
  }
}
}  // namespace elman
