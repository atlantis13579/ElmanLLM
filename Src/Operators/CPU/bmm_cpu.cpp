#include "kernel.h"

namespace elman {
void launch_bmm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int rows = p.output_shape(p.output_rank() - 2);
    int columns = p.output_shape(p.output_rank() - 1);
    int row = (i / columns) % rows;
    int column = i % columns;
    int batch = i / (rows * columns);
    int abase = matrix_batch_offset(batch, KernelInput::A, p);
    int bbase = matrix_batch_offset(batch, KernelInput::B, p);
    int shared_size =
        p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - 1);
    float value = 0.0f;
    for (int k = 0; k < shared_size; ++k) {
      value += READ_A(abase + row * shared_size + k) *
               READ_B(bbase + k * columns + column);
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
