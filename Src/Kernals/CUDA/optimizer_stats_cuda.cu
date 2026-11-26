#include "kernel.cuh"

namespace elman {
namespace {
__global__ void optimizer_stats_kernel(float* output, const float* weights, const float* gradients,
                                     const float* first, const float* second, int count, bool norm) {
  __shared__ float sums[8];
  __shared__ unsigned errors[8];
  const int thread = int(threadIdx.x), start = int(blockIdx.x) * 4096;
  float square = 0;
  unsigned invalid = 0;
  for (int i = start + thread; i < count && i < start + 4096; i += 256) {
    invalid |= !isfinite(weights[i]) || !isfinite(first[i]) || !isfinite(second[i]) || second[i] < 0;
    if (norm) { const float g = gradients[i]; square += g * g; invalid |= !isfinite(g); }
  }
  for (int offset = 16; offset; offset /= 2) {
    square += __shfl_down_sync(0xffffffff, square, offset);
    invalid |= __shfl_down_sync(0xffffffff, invalid, offset);
  }
  if (thread % 32 == 0) { sums[thread / 32] = square; errors[thread / 32] = invalid; }
  __syncthreads();
  if (thread < 32) {
    square = thread < 8 ? sums[thread] : 0.f;
    invalid = thread < 8 ? errors[thread] : 0;
    for (int offset = 16; offset; offset /= 2) {
      square += __shfl_down_sync(0xffffffff, square, offset);
      invalid |= __shfl_down_sync(0xffffffff, invalid, offset);
    }
    if (!thread) { output[blockIdx.x * 2] = square; output[blockIdx.x * 2 + 1] = float(invalid); }
  }
}
}
void launch_optimizer_stats_cuda(uint32_t* output, const uint32_t* weights, const uint32_t* gradients,
                                 const uint32_t* first, const uint32_t* second, const KernelParams& p) {
  int count = 1;
  for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count *= p.input_shape(KernelInput::A, d);
  optimizer_stats_kernel<<<p.output_shape(0), 256, 0, current_cuda_stream()>>>(reinterpret_cast<float*>(output),
      reinterpret_cast<const float*>(weights), reinterpret_cast<const float*>(gradients),
      reinterpret_cast<const float*>(first), reinterpret_cast<const float*>(second), count, p.attribute(0) != 0);
}
} // namespace elman
