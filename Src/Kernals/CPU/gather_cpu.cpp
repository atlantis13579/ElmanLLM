#include "kernel.h"

namespace elman {
void launch_gather_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int flat = i;
    int source = 0;
    int selected = READ_INT_B(i);
    int valid =
        selected >= 0 &&
        selected < p.input_shape(KernelInput::A,
                                 p.attribute(kernal_attr::indexing::kAxis));
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      if (axis == p.attribute(kernal_attr::indexing::kAxis)) {
        coordinate = selected;
      }
      source += coordinate * p.input_stride(KernelInput::A, axis);
    }
    if (valid != 0) {
      COPY_INPUT_A(i, source);
    } else {
      WRITE_OUTPUT(i, 0.0f);
    }
  }
}
}  // namespace elman
