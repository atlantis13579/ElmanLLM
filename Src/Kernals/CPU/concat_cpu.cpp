#include "kernel.h"

namespace elman {
void launch_concat_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int axis = p.attribute(kernal_attr::concat::kAxis);
    int inner = p.input_stride(KernelInput::A, axis);
    int full = p.output_shape(axis);
    int coordinate = (i / inner) % full;
    int outer = i / (inner * full);
    int rest = i % inner;
    if (coordinate < p.input_shape(KernelInput::A, axis)) {
      int source = outer * p.input_shape(KernelInput::A, axis) * inner +
                   coordinate * inner + rest;
      COPY_INPUT_A(i, source);
    } else {
      int source = outer * p.input_shape(KernelInput::B, axis) * inner +
                   (coordinate - p.input_shape(KernelInput::A, axis)) * inner +
                   rest;
      if (p.input_dtype(KernelInput::B) == p.output_dtype()) {
        COPY_TYPED_ELEMENT(out, b, i, source, p.output_dtype());
      } else if (!kernel_integer(p.input_dtype(KernelInput::B))) {
        WRITE_OUTPUT(i, READ_B(source));
      } else {
        WRITE_INT_OUTPUT(i, READ_INT_B(source));
      }
    }
  }
}
}  // namespace elman
