#include "kernel.cuh"

namespace elman {
__global__ void kernel_where_cuda(uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c,
                                  KernelParams p) {
  const unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    const int i = int(element);
    const bool select = READ_INT_A(broadcast_index(i, KernelInput::A, p)) != 0;
    const int index =
        broadcast_index(i, select ? KernelInput::B : KernelInput::C, p);
    COPY_TYPED_ELEMENT(out, select ? b : c, i, index, p.output_dtype());
  }
}
void launch_where_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_where_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
