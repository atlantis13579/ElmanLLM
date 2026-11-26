#pragma once
#include <cuda_runtime.h>

namespace elman {
// Each row uses a full block, including when vocabulary size exceeds block
// width.
template <bool Maximum, class Float>
__device__ Float row_reduce(Float value, Float* scratch) {
  scratch[threadIdx.x] = value;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride; stride /= 2) {
    if (threadIdx.x < stride) {
      scratch[threadIdx.x] =
          Maximum ? fmax(scratch[threadIdx.x], scratch[threadIdx.x + stride])
                  : scratch[threadIdx.x] + scratch[threadIdx.x + stride];
    }
    __syncthreads();
  }
  Float result = scratch[0];
  __syncthreads();
  return result;
}
}  // namespace elman
