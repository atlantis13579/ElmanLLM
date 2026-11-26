#include "kernel.h"

namespace elman {
void launch_repeat_backward_cpu(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float value = 0.0f;
    int count = input_count(KernelInput::A, p);
    int delta = p.input_rank(KernelInput::A) - p.output_rank();
    for (int j = 0; j < count; ++j) {
      int flat = j;
      int target = 0;
      int stride = 1;
      for (int axis = p.input_rank(KernelInput::A) - 1; axis >= 0; --axis) {
        int coordinate = flat % p.input_shape(KernelInput::A, axis);
        flat /= p.input_shape(KernelInput::A, axis);
        int target_axis = axis - delta;
        if (target_axis >= 0) {
          target += (coordinate % p.output_shape(target_axis)) * stride;
          stride *= p.output_shape(target_axis);
        }
      }
      if (target == i) {
        value += READ_A(j);
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
