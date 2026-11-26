#include "kernel.cuh"

namespace elman {
__global__ void kernel_cast_cuda(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c,
                                 KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    if (p.input_dtype(KernelInput::A) == p.output_dtype()) {
      COPY_INPUT_A(i, i);
      return;
    }
    if (kernel_integer(p.input_dtype(KernelInput::A))) {
      WRITE_INT_OUTPUT(i, READ_INT_A(i));
      return;
    }
    float x = READ_A(i);
    WRITE_OUTPUT(i, x);
  }
}
void launch_cast_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_cast_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
