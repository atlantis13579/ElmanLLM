#include "kernel.cuh"

namespace elman {
__global__ void kernel_tril_cuda(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c,
                                 KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int columns = p.output_shape(p.output_rank() - 1);
    int rows = p.output_shape(p.output_rank() - 2);
    int row = (i / columns) % rows;
    int column = i % columns;
    if (column - row <= p.attribute(operator_attr::triangular::kDiagonal)) {
      COPY_INPUT_A(i, i);
    } else {
      WRITE_OUTPUT(i, 0.0f);
    }
  }
}
void launch_tril_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_tril_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
