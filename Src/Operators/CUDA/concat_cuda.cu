#include "kernel.cuh"

namespace elman {
__global__ void kernel_concat_cuda(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(operator_attr::concat::kAxis);
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
      if (p.input_dtype(KernelInput::B) == 0) {
        WRITE_OUTPUT(i, READ_B(source));
      } else {
        WRITE_INT_OUTPUT(i, READ_INT_B(source));
      }
    }
  }
}
void launch_concat_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_concat_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
