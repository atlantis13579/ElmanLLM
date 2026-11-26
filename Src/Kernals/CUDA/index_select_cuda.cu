#include "kernel.cuh"

namespace elman {
__global__ void kernel_index_select_cuda(uint32_t* out, const uint32_t* a,
                                         const uint32_t* b, const uint32_t* c,
                                         KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(operator_attr::indexing::kAxis);
    int inner = p.input_stride(KernelInput::A, axis);
    int output_length = p.output_shape(axis);
    int selected = READ_INT_B((i / inner) % output_length);
    int source = (i / (inner * output_length)) *
                     p.input_shape(KernelInput::A, axis) * inner +
                 selected * inner + i % inner;
    if (selected >= 0 && selected < p.input_shape(KernelInput::A, axis)) {
      COPY_INPUT_A(i, source);
    } else {
      WRITE_OUTPUT(i, 0.0f);
    }
  }
}
void launch_index_select_cuda(uint32_t* out, const uint32_t* a,
                              const uint32_t* b, const uint32_t* c,
                              const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_index_select_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                               256, 0, current_cuda_stream()>>>(out, a, b, c, p);
  }
}
}  // namespace elman
