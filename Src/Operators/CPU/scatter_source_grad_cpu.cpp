#include "kernel.h"

namespace elman {
void launch_scatter_source_grad_cpu(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int flat = i;
    int target = 0;
    int selected = READ_INT_B(i);
    int valid =
        selected >= 0 &&
        selected < p.input_shape(KernelInput::A,
                                 p.attribute(operator_attr::indexing::kAxis));
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      if (axis == p.attribute(operator_attr::indexing::kAxis)) {
        coordinate = selected;
      }
      target += coordinate * p.input_stride(KernelInput::A, axis);
    }
    WRITE_OUTPUT(i, valid != 0 ? READ_A(target) : 0.0f);
  }
}
}  // namespace elman
