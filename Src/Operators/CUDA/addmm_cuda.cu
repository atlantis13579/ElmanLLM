#include "kernel.cuh"
#include "Backends/CudaGemm.h"

namespace elman {
__global__ void kernel_addmm_cuda(uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c,
                                  KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int rows = p.output_shape(p.output_rank() - 2);
    int columns = p.output_shape(p.output_rank() - 1);
    int row = (i / columns) % rows;
    int column = i % columns;
    int batch = i / (rows * columns);
    int abase = matrix_batch_offset(batch, KernelInput::A, p);
    int bbase = matrix_batch_offset(batch, KernelInput::B, p);
    int shared_size =
        p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - 1);
    if (kernel_integer(p.output_dtype())) {
      uint32_t integer_value = 0u;
      for (int k = 0; k < shared_size; ++k) {
        integer_value += uint32_t(READ_INT_A(abase + row * shared_size + k)) *
                         uint32_t(READ_INT_B(bbase + k * columns + column));
      }
      integer_value =
          uint32_t(int(float_attribute(p, operator_attr::addmm::kAlpha))) *
              integer_value +
          uint32_t(int(float_attribute(p, operator_attr::addmm::kBeta))) *
              uint32_t(READ_INT_C(broadcast_index(i, KernelInput::C, p)));
      WRITE_INT_OUTPUT(i, int(integer_value));
      return;
    }
    float value = 0.0f;
    const float alpha = float_attribute(p, operator_attr::addmm::kAlpha);
    const float beta = float_attribute(p, operator_attr::addmm::kBeta);
    for (int k = 0; alpha != 0.0f && k < shared_size; ++k) {
      value += READ_A(abase + row * shared_size + k) *
               READ_B(bbase + k * columns + column);
    }
    WRITE_OUTPUT(i, alpha * value +
                        (beta == 0.0f ? 0.0f : beta * READ_C(broadcast_index(i, KernelInput::C, p))));
  }
}
void launch_addmm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  if (try_gemm_cuda(out, a, b, c, p, GemmOperation::Addmm)) return;
  if (p.output_count() > 0) {
    kernel_addmm_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
