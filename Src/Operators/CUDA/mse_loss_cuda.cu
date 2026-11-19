#include "Core/AllocationPool.h"
#include "kernel.cuh"
#include "row_reduce.cuh"

#include <algorithm>

namespace elman {
namespace {
constexpr int threads = 256;

__global__ void squared_error(const uint32_t* a, const uint32_t* b, float* partial,
                             uint32_t* out, int count, KernelParams p) {
  __shared__ float scratch[threads];
  float sum = 0;
  for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
       i += int64_t(gridDim.x) * blockDim.x) {
    const float diff = READ_A(int(i)) - READ_B(int(i));
    sum += __fmul_rn(diff, diff);
  }
  sum = row_reduce<false>(sum, scratch);
  if (threadIdx.x == 0) {
    if (partial) partial[blockIdx.x] = sum;
    else WRITE_OUTPUT(0, sum / float(count));
  }
}

__global__ void finish_squared_error(uint32_t* out, const float* partial, int blocks,
                                    int count, KernelParams p) {
  __shared__ float scratch[threads];
  float sum = 0;
  for (int i = threadIdx.x; i < blocks; i += threads) sum += partial[i];
  sum = row_reduce<false>(sum, scratch);
  if (threadIdx.x == 0) WRITE_OUTPUT(0, sum / float(count));
}
}  // namespace

void launch_mse_loss_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const uint32_t*, const KernelParams& p) {
  if (!p.output_count()) return;
  int count = 1;
  for (int axis = 0; axis < p.input_rank(KernelInput::A); ++axis)
    count *= p.input_shape(KernelInput::A, axis);
  const int blocks = std::clamp((count / (threads * 4)) + 1, 1, 256);
  const auto stream = current_cuda_stream();
  if (blocks == 1) {
    squared_error<<<1, threads, 0, stream>>>(a, b, nullptr, out, count, p);
    return;
  }
  auto workspace = backend_workspace(Backend::CUDA, size_t(blocks) * sizeof(float));
  auto* partial = static_cast<float*>(workspace->raw_pointer());
  squared_error<<<blocks, threads, 0, stream>>>(a, b, partial, nullptr, count, p);
  finish_squared_error<<<1, threads, 0, stream>>>(out, partial, blocks, count, p);
}
}  // namespace elman
