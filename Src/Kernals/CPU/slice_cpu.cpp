#include "kernel.h"

namespace elman {
void launch_slice_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int flat = i;
    int source = 0;
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      if (axis == p.attribute(kernal_attr::slice::kAxis)) {
        coordinate = p.attribute(kernal_attr::slice::kStart) +
                     coordinate * p.attribute(kernal_attr::slice::kStep);
      }
      source += coordinate * p.input_stride(KernelInput::A, axis);
    }
    COPY_INPUT_A(i, source);
  }
}
}  // namespace elman
