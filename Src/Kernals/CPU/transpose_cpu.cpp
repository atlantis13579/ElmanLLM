#include "kernel.h"

namespace elman {
void launch_transpose_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int flat = i;
    int source = 0;
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      int source_axis =
          axis == p.attribute(kernal_attr::transpose::kFirstAxis)
              ? p.attribute(kernal_attr::transpose::kSecondAxis)
              : (axis == p.attribute(kernal_attr::transpose::kSecondAxis)
                     ? p.attribute(kernal_attr::transpose::kFirstAxis)
                     : axis);
      source += coordinate * p.input_stride(KernelInput::A, source_axis);
    }
    COPY_INPUT_A(i, source);
  }
}
}  // namespace elman
