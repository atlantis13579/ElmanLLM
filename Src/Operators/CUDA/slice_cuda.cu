#include "kernel.cuh"

namespace elman {
__global__ void kernel_slice_cuda(uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c,
                                  KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int flat = i;
    int source = 0;
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      if (axis == p.attribute(operator_attr::slice::kAxis)) {
        coordinate = p.attribute(operator_attr::slice::kStart) +
                     coordinate * p.attribute(operator_attr::slice::kStep);
      }
      source += coordinate * p.input_stride(KernelInput::A, axis);
    }
    COPY_INPUT_A(i, source);
  }
}
void launch_slice_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_slice_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
