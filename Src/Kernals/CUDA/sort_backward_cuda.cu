#include "kernel.cuh"
#include "Kernals/sort_params.h"

namespace elman {
__global__ void sort_backward_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                   KernelParams p, int length, int inner) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= p.output_count()) return;
  int target = i / (length * inner) * length * inner + i % inner + READ_INT_B(i) * inner;
  WRITE_OUTPUT(target, READ_A(i));
}
void launch_sort_backward_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                               const uint32_t*, const KernelParams& p) {
  if (!p.output_count()) return;
  int axis = p.attribute(operator_attr::sorting::kAxis);
  sort_backward_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, a, b, p,
      p.output_shape(axis), p.input_stride(KernelInput::A, axis));
}
} // namespace elman
