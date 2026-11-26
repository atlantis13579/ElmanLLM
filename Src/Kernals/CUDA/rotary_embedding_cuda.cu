#include "kernel.cuh"

namespace elman {
__global__ void kernel_rotary_embedding_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                             const uint32_t* c, KernelParams p) {
  unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= unsigned(p.output_count())) return;
  int i = int(element), width = p.output_shape(p.output_rank() - 1), half = width / 2, dtype = p.output_dtype();
  float rotated = i % width < half ? -READ_A(i + half) : READ_A(i - half);
  float left = __fmul_rn(READ_A(i), READ_B(broadcast_index(i, KernelInput::B, p)));
  float right = __fmul_rn(rotated, READ_C(broadcast_index(i, KernelInput::C, p)));
  left = elman_cuda_decode(elman_cuda_encode(left, dtype), dtype);
  right = elman_cuda_decode(elman_cuda_encode(right, dtype), dtype);
  WRITE_OUTPUT(i, __fadd_rn(left, right));
}
void launch_rotary_embedding_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                  const uint32_t* c, const KernelParams& p) {
  if (p.output_count()) kernel_rotary_embedding_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, a, b, c, p);
}
} // namespace elman
