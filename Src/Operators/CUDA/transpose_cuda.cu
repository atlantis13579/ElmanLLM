#include "kernel.cuh"

namespace elman {
__global__ void kernel_transpose_cuda(uint32_t* out, const uint32_t* a,
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
      int source_axis =
          axis == p.attribute(operator_attr::transpose::kFirstAxis)
              ? p.attribute(operator_attr::transpose::kSecondAxis)
              : (axis == p.attribute(operator_attr::transpose::kSecondAxis)
                     ? p.attribute(operator_attr::transpose::kFirstAxis)
                     : axis);
      source += coordinate * p.input_stride(KernelInput::A, source_axis);
    }
    COPY_INPUT_A(i, source);
  }
}
void launch_transpose_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_transpose_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
