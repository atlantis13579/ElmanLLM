#include "kernel.h"

namespace elman {
void launch_index_select_backward_cpu(uint32_t* out, const uint32_t* a,
                                      const uint32_t* b, const uint32_t* c,
                                      const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int axis = p.attribute(kernal_attr::indexing::kAxis);
    int inner = output_stride(axis, p);
    int length = p.output_shape(axis);
    int selected = (i / inner) % length;
    int count = input_count(KernelInput::B, p);
    float value = 0.0f;
    for (int j = 0; j < count; ++j) {
      if (READ_INT_B(j) == selected) {
        value += READ_A((i / (length * inner)) * count * inner + j * inner +
                        i % inner);
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
