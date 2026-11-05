#include "kernel.cuh"

namespace elman {
__global__ void kernel_repeat_grad_cuda(uint32_t* out, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c,
                                        KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float value = 0.0f;
    int count = input_count(KernelInput::A, p);
    int delta = p.input_rank(KernelInput::A) - p.output_rank();
    for (int j = 0; j < count; ++j) {
      int flat = j;
      int target = 0;
      int stride = 1;
      for (int axis = p.input_rank(KernelInput::A) - 1; axis >= 0; --axis) {
        int coordinate = flat % p.input_shape(KernelInput::A, axis);
        flat /= p.input_shape(KernelInput::A, axis);
        int target_axis = axis - delta;
        if (target_axis >= 0) {
          target += (coordinate % p.output_shape(target_axis)) * stride;
          stride *= p.output_shape(target_axis);
        }
      }
      if (target == i) {
        value += READ_A(j);
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_repeat_grad_cuda(uint32_t* out, const uint32_t* a,
                             const uint32_t* b, const uint32_t* c,
                             const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_repeat_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                              256>>>(out, a, b, c, p);
  }
}
}  // namespace elman
