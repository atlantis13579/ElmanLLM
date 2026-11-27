#include "kernel.h"

namespace elman {
void launch_scatter_add_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    float value = READ_A(i);
    uint32_t integer_value = kernel_integer(p.output_dtype()) ? uint32_t(READ_INT_A(i)) : 0u;
    int count = input_count(KernelInput::C, p);
    for (int j = 0; j < count; ++j) {
      int flat = j;
      int target = 0;
      int selected = READ_INT_B(j);
      for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
        int coordinate = flat % p.input_shape(KernelInput::C, axis);
        flat /= p.input_shape(KernelInput::C, axis);
        if (axis == p.attribute(kernal_attr::indexing::kAxis)) {
          coordinate = selected;
        }
        target += coordinate * p.input_stride(KernelInput::A, axis);
      }
      if (target == i && selected >= 0 &&
          selected <
              p.output_shape(p.attribute(kernal_attr::indexing::kAxis))) {
        value += READ_C(j);
        if (kernel_integer(p.output_dtype())) integer_value += uint32_t(READ_INT_C(j));
      }
    }
    if (!kernel_integer(p.output_dtype())) {
      WRITE_OUTPUT(i, value);
    } else {
      WRITE_INT_OUTPUT(i, int(integer_value));
    }
  }
}
}  // namespace elman
