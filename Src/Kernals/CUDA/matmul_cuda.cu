#include "kernel.cuh"
#include "Backends/CudaGemm.h"
#include "Kernals/matmul_params.h"

namespace elman {
__global__ void kernel_matmul_cuda(uint32_t* out, const uint32_t* a,
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
    const bool transposed_a = p.attribute(operator_attr::matmul::kTransposeA) != 0;
    int shared_size =
        p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - (transposed_a ? 2 : 1));
    bool transposed = p.attribute(operator_attr::matmul::kTransposeB) != 0;
    if (kernel_integer(p.output_dtype())) {
      uint32_t integer_value = 0u;
      for (int k = 0; k < shared_size; ++k) {
        integer_value += uint32_t(READ_INT_A(abase + row * shared_size + k)) *
                         uint32_t(READ_INT_B(bbase + (transposed ? column * shared_size + k : k * columns + column)));
      }
      WRITE_INT_OUTPUT(i, int(integer_value));
      return;
    }
    float value = 0.0f;
    for (int k = 0; k < shared_size; ++k) {
      value += READ_A(abase + (transposed_a ? k * rows + row : row * shared_size + k)) *
               READ_B(bbase + (transposed ? column * shared_size + k : k * columns + column));
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_matmul_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (try_gemm_cuda(out, a, b, c, p, GemmOperation::Matmul)) return;
  if (p.output_count() > 0) {
    kernel_matmul_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
