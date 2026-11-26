#include "kernel.h"

namespace elman {
void launch_repeat_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int flat = i;
    int source = 0;
    int delta = p.output_rank() - p.input_rank(KernelInput::A);
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      int source_axis = axis - delta;
      if (source_axis >= 0) {
        source += (coordinate % p.input_shape(KernelInput::A, source_axis)) *
                  p.input_stride(KernelInput::A, source_axis);
      }
    }
    COPY_INPUT_A(i, source);
  }
}
}  // namespace elman
