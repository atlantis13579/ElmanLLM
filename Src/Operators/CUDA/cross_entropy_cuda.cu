#include <stdexcept>

#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "row_reduce.cuh"

namespace elman {
__global__ void cross_entropy_rows(float* losses, const uint32_t* a,
                                   const uint32_t* b, KernelParams p) {
  __shared__ float scratch[256];
  int row = int(blockIdx.x),
      columns = p.input_shape(KernelInput::A, tensor_axis::logits::kClasses);
  float highest = -EL_INF;
  for (int column = int(threadIdx.x); column < columns;
       column += int(blockDim.x)) {
    highest = fmaxf(highest, READ_A(row * columns + column));
  }
  highest = row_reduce<true>(highest, scratch);
  float denominator = 0;
  for (int column = int(threadIdx.x); column < columns;
       column += int(blockDim.x)) {
    denominator += expf(READ_A(row * columns + column) - highest);
  }
  denominator = row_reduce<false>(denominator, scratch);
  if (threadIdx.x == 0) {
    int target = READ_INT_B(row);
    losses[row] =
        target >= 0 && target < columns
            ? logf(denominator) + (highest - READ_A(row * columns + target))
            : EL_INF;
  }
}
__global__ void cross_entropy_average(uint32_t* out, const float* losses,
                                      KernelParams p) {
  __shared__ float scratch[256];
  float value = 0;
  for (int row = int(threadIdx.x);
       row < p.input_shape(KernelInput::A, tensor_axis::logits::kBatch);
       row += int(blockDim.x)) {
    value += losses[row];
  }
  value = row_reduce<false>(value, scratch);
  if (threadIdx.x == 0) {
    WRITE_OUTPUT(0, value / float(p.input_shape(KernelInput::A,
                                                tensor_axis::logits::kBatch)));
  }
}
void launch_cross_entropy_cuda(uint32_t* out, const uint32_t* a,
                               const uint32_t* b, const uint32_t*,
                               const KernelParams& p) {
  float* losses = nullptr;
  if (p.input_shape(KernelInput::A, tensor_axis::logits::kBatch) == 0) {
    cross_entropy_average<<<1, 256, 0, current_cuda_stream()>>>(out, losses, p);
    return;
  }
  auto storage = backend_workspace(Backend::CUDA,
      size_t(p.input_shape(KernelInput::A, tensor_axis::logits::kBatch)) * sizeof(float));
  losses = static_cast<float*>(storage->raw_pointer());
  cross_entropy_rows<<<
      p.input_shape(KernelInput::A, tensor_axis::logits::kBatch), 256, 0, current_cuda_stream()>>>(
      losses, a, b, p);
  cross_entropy_average<<<1, 256, 0, current_cuda_stream()>>>(out, losses, p);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}
}  // namespace elman
