#include "kernel.h"

namespace elman {
void launch_index_select_cpu(uint32_t* out, const uint32_t* a,
                             const uint32_t* b, const uint32_t* c,
                             const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int axis = p.attribute(kernal_attr::indexing::kAxis);
    int inner = p.input_stride(KernelInput::A, axis);
    int output_length = p.output_shape(axis);
    int selected = READ_INT_B((i / inner) % output_length);
    int source = (i / (inner * output_length)) *
                     p.input_shape(KernelInput::A, axis) * inner +
                 selected * inner + i % inner;
    if (selected >= 0 && selected < p.input_shape(KernelInput::A, axis)) {
      COPY_INPUT_A(i, source);
    } else {
      WRITE_OUTPUT(i, 0.0f);
    }
  }
}
}  // namespace elman
