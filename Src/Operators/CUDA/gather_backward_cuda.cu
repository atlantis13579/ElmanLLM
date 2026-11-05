#include "kernel.cuh"

namespace elman {
__global__ void kernel_gather_backward_cuda(uint32_t* out, const uint32_t* a,
                                            const uint32_t* b,
                                            const uint32_t* c, KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float value = 0.0f;
    int count = input_count(KernelInput::A, p);
    for (int j = 0; j < count; ++j) {
      int flat = j;
      int target = 0;
      int stride = 1;
      int selected = READ_INT_B(j);
      for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
        int coordinate = flat % p.input_shape(KernelInput::A, axis);
        flat /= p.input_shape(KernelInput::A, axis);
        if (axis == p.attribute(operator_attr::indexing::kAxis)) {
          coordinate = selected;
        }
        target += coordinate * stride;
        stride *= p.output_shape(axis);
      }
      if (target == i && selected >= 0 &&
          selected <
              p.output_shape(p.attribute(operator_attr::indexing::kAxis))) {
        value += READ_A(j);
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_gather_backward_cuda(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c,
                                 const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_gather_backward_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                                  256>>>(out, a, b, c, p);
  }
}
}  // namespace elman
