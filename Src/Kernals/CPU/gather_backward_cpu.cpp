#include "kernel.h"

namespace elman {
void launch_gather_backward_cpu(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float value = 0.0f;
    int count = input_count(KernelInput::A, p);
    for (int j = 0; j < count; ++j) {
      int flat = j;
      int target = 0;
      int stride = 1;
      int selected = READ_INT_B(j);
      for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
        int coordinate = flat % p.input_shape(KernelInput::A, axis);
        flat /= p.input_shape(KernelInput::A, axis);
        if (axis == p.attribute(kernal_attr::indexing::kAxis)) {
          coordinate = selected;
        }
        target += coordinate * stride;
        stride *= p.output_shape(axis);
      }
      if (target == i && selected >= 0 &&
          selected <
              p.output_shape(p.attribute(kernal_attr::indexing::kAxis))) {
        value += READ_A(j);
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
