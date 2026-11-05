#include "kernel.cuh"

namespace elman {
__global__ void kernel_index_select_grad_cuda(uint32_t* out, const uint32_t* a,
                                              const uint32_t* b,
                                              const uint32_t* c,
                                              KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(operator_attr::indexing::kAxis);
    int inner = output_stride(axis, p);
    int length = p.output_shape(axis);
    int selected = (i / inner) % length;
    int count = input_count(KernelInput::B, p);
    float value = 0.0f;
    for (int j = 0; j < count; ++j) {
      if (READ_INT_B(j) == selected) {
        value += READ_A((i / (length * inner)) * count * inner + j * inner +
                        i % inner);
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_index_select_grad_cuda(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_index_select_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                                    256>>>(out, a, b, c, p);
  }
}
}  // namespace elman
