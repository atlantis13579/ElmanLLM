#include "kernel.cuh"

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
    float value = 0.0f;
    for (int k = 0; k < shared_size; ++k) {
      value += READ_A(abase + row * shared_size + k) *
               READ_B(bbase + k * columns + column);
    }
    WRITE_OUTPUT(i, float_attribute(p, operator_attr::addmm::kAlpha) * value +
                        float_attribute(p, operator_attr::addmm::kBeta) *
                            READ_C(broadcast_index(i, KernelInput::C, p)));
  }
}
void launch_addmm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_addmm_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
