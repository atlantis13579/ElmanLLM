#include "kernel.cuh"

namespace elman {
__global__ void kernel_gather_cuda(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int flat = i;
    int source = 0;
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
      source += coordinate * p.input_stride(KernelInput::A, axis);
    }
    if (valid != 0) {
      COPY_INPUT_A(i, source);
    } else {
      WRITE_OUTPUT(i, 0.0f);
    }
  }
}
void launch_gather_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_gather_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
