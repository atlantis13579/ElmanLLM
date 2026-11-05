#include "kernel.cuh"

namespace elman {
__global__ void kernel_scatter_add_cuda(uint32_t* out, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c,
                                        KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    float value = READ_A(i);
    int integer_value = READ_INT_A(i);
    int count = input_count(KernelInput::C, p);
    for (int j = 0; j < count; ++j) {
      int flat = j;
      int target = 0;
      int selected = READ_INT_B(j);
      for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
        int coordinate = flat % p.input_shape(KernelInput::C, axis);
        flat /= p.input_shape(KernelInput::C, axis);
        if (axis == p.attribute(operator_attr::indexing::kAxis)) {
          coordinate = selected;
        }
        target += coordinate * p.input_stride(KernelInput::A, axis);
      }
      if (target == i && selected >= 0 &&
          selected <
              p.output_shape(p.attribute(operator_attr::indexing::kAxis))) {
        value += READ_C(j);
        integer_value += READ_INT_C(j);
      }
    }
    if (p.output_dtype() == 0) {
      WRITE_OUTPUT(i, value);
    } else {
      WRITE_INT_OUTPUT(i, integer_value);
    }
  }
}
void launch_scatter_add_cuda(uint32_t* out, const uint32_t* a,
                             const uint32_t* b, const uint32_t* c,
                             const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_scatter_add_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                              256>>>(out, a, b, c, p);
  }
}
}  // namespace elman
