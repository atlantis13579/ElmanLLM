#include "kernel.cuh"
#include "Backends/CudaGemm.h"

namespace elman {
__global__ void kernel_linear_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                  const uint32_t* c, KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    const int outputs = p.input_shape(KernelInput::B, 0);
    const int inputs = p.input_shape(KernelInput::B, 1);
    const int row = i / outputs, column = i % outputs;
    if (kernel_integer(p.output_dtype())) {
      uint32_t value = 0;
      for (int k = 0; k < inputs; ++k)
        value += uint32_t(READ_INT_A(row * inputs + k)) * uint32_t(READ_INT_B(column * inputs + k));
      if (p.input_rank(KernelInput::C) >= 0) value += uint32_t(READ_INT_C(p.input_rank(KernelInput::C) == 0 ? 0 : column));
      WRITE_INT_OUTPUT(i, int32_t(value));
    } else {
      float value = 0;
      for (int k = 0; k < inputs; ++k) value += READ_A(row * inputs + k) * READ_B(column * inputs + k);
      if (p.input_rank(KernelInput::C) >= 0) value += READ_C(p.input_rank(KernelInput::C) == 0 ? 0 : column);
      WRITE_OUTPUT(i, value);
    }
  }
}
void launch_linear_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (try_gemm_cuda(out, a, b, c, p, GemmOperation::Linear)) return;
  if (p.output_count() > 0)
    kernel_linear_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, a, b, c, p);
}
} // namespace elman
