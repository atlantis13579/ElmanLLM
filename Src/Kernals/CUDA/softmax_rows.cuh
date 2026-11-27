#pragma once
#include "kernel.cuh"
#include "softmax_large.cuh"
#include <algorithm>
#include <stdexcept>

namespace elman {
template <int Power, class Input> __global__ void softmax_rows_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                                      KernelParams p, int length, int rows, Input read) {
  constexpr int width = Power < 32 ? Power : 32;
  constexpr int iterations = Power / width;
  int row = int(blockIdx.x * blockDim.y + threadIdx.y), lane = int(threadIdx.x);
  float values[iterations], maximum = -EL_INF;
  #pragma unroll
  for (int j = 0; j < iterations; ++j) {
    int column = lane + j * width;
    values[j] = row < rows && column < length ? read(a, b, p, row * length + column) : -EL_INF;
    maximum = EL_MAX(maximum, values[j]);
  }
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    float other = __shfl_xor_sync(0xffffffff, maximum, offset, width);
    maximum = EL_MAX(maximum, other);
  }
  float denominator = 0;
  #pragma unroll
  for (int j = 0; j < iterations; ++j) {
    values[j] = EL_EXP(values[j] - maximum);
    denominator += values[j];
  }
  for (int offset = width / 2; offset > 0; offset >>= 1)
    denominator += __shfl_xor_sync(0xffffffff, denominator, offset, width);
  #pragma unroll
  for (int j = 0; j < iterations; ++j) {
    int column = lane + j * width;
    if (row < rows && column < length) WRITE_OUTPUT(row * length + column, values[j] / denominator);
  }
}
template <int Power, class Input> void launch_softmax_rows(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                             const KernelParams& p, int length, Input read) {
  constexpr int width = Power < 32 ? Power : 32;
  int rows = p.output_count() / length;
  // Small decode batches leave most SMs idle if four long rows share each CTA.
  const int rows_per_block = rows <= 32 && length >= 256 ? 1 : 128 / width;
  softmax_rows_cuda<Power><<<(rows + rows_per_block - 1) / rows_per_block, dim3(width, rows_per_block), 0, current_cuda_stream()>>>(out, a, b, p, length, rows, read);
}
// Adjacent x lanes visit contiguous inner coordinates. The y lanes share
// one softmax reduction, so each input is visited a constant number of times.
template<class Input> __global__ void spatial_softmax_cuda(uint32_t* out, const uint32_t* a,
        const uint32_t* b, KernelParams p, int inner, int length, int tiles, Input read) {
  __shared__ float partial[256];
  const int lane = threadIdx.y * blockDim.x + threadIdx.x;
  const int column = (blockIdx.x % tiles) * blockDim.x + threadIdx.x;
  const bool active = column < inner;
  const int base = active ? int(blockIdx.x / tiles) * length * inner + column : 0;
  float maximum = -EL_INF;
  if (active) for (int j = threadIdx.y; j < length; j += blockDim.y)
    maximum = EL_MAX(maximum, read(a, b, p, base + j * inner));
  partial[lane] = maximum;
  __syncthreads();
  for (unsigned step = blockDim.y / 2; step; step /= 2) {
    if (threadIdx.y < step) partial[lane] = EL_MAX(partial[lane], partial[lane + step * blockDim.x]);
    __syncthreads();
  }
  maximum = partial[threadIdx.x];
  float sum = 0;
  if (active) for (int j = threadIdx.y; j < length; j += blockDim.y)
    sum += EL_EXP(read(a, b, p, base + j * inner) - maximum);
  __syncthreads();
  partial[lane] = sum;
  __syncthreads();
  for (unsigned step = blockDim.y / 2; step; step /= 2) {
    if (threadIdx.y < step) partial[lane] += partial[lane + step * blockDim.x];
    __syncthreads();
  }
  sum = partial[threadIdx.x];
  if (active) for (int j = threadIdx.y; j < length; j += blockDim.y) {
    const int i = base + j * inner;
    WRITE_OUTPUT(i, EL_EXP(read(a, b, p, i) - maximum) / sum);
  }
}
template<class Input> void launch_softmax_cuda_impl(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                                    const KernelParams& p, Input read, int input_dtype) {
  if (p.output_count() > 0) {
    int axis = p.attribute(kernal_attr::softmax::kAxis);
    int inner = p.input_rank(KernelInput::A) == 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = p.input_rank(KernelInput::A) == 0 ? 1 : p.input_shape(KernelInput::A, axis);
    if (inner == 1 && length <= 2048) {
      int power = 1;
      while (power < length) power *= 2;
      #define ELMAN_SOFTMAX_POWER(n) case n: launch_softmax_rows<n>(out, a, b, p, length, read); return;
      switch (power) {
        ELMAN_SOFTMAX_POWER(1) ELMAN_SOFTMAX_POWER(2) ELMAN_SOFTMAX_POWER(4) ELMAN_SOFTMAX_POWER(8)
        ELMAN_SOFTMAX_POWER(16) ELMAN_SOFTMAX_POWER(32) ELMAN_SOFTMAX_POWER(64) ELMAN_SOFTMAX_POWER(128)
        ELMAN_SOFTMAX_POWER(256) ELMAN_SOFTMAX_POWER(512) ELMAN_SOFTMAX_POWER(1024) ELMAN_SOFTMAX_POWER(2048)
      }
      #undef ELMAN_SOFTMAX_POWER
    }
    if (inner == 1) {
      int device = 0, shared_bytes = 0;
      if (cudaGetDevice(&device) != cudaSuccess ||
          cudaDeviceGetAttribute(&shared_bytes, cudaDevAttrMaxSharedMemoryPerBlock, device) != cudaSuccess)
        throw std::runtime_error("Cannot query CUDA shared memory for softmax");
      int bytes = input_dtype == int(DType::Float32) ? 4 : 2;
      int width = 16 / bytes;
      int path = (length + 1023) / 1024 < 10 ? 0 :
                 (length % width == 0 && size_t(length) * bytes + 128 < size_t(shared_bytes) ? 1 : 2);
      int rows = p.output_count() / length;
      if (width == 4) softmax_large_rows<4><<<rows, 1024, 0, current_cuda_stream()>>>(out, a, b, p, length, path, read);
      else softmax_large_rows<8><<<rows, 1024, 0, current_cuda_stream()>>>(out, a, b, p, length, path, read);
      return;
    }
    unsigned columns = 1, segments = 1;
    // Long reductions with a small inner dimension need several CTAs across
    // that dimension to keep the device occupied.
    const unsigned rows = unsigned(p.output_count() / length);
    const unsigned column_limit = length < 64 ? 32 : std::clamp(rows / 32u, 1u, 32u);
    while (columns * 2 <= column_limit && columns < unsigned(inner)) columns *= 2;
    while (segments < 256 / columns && segments < unsigned(length)) segments *= 2;
    const unsigned tiles = (unsigned(inner) + columns - 1) / columns;
    const unsigned outer = unsigned(p.output_count() / inner / length);
    spatial_softmax_cuda<<<outer * tiles, dim3(columns, segments), 0, current_cuda_stream()>>>(
        out, a, b, p, inner, length, tiles, read);
  }
}
}  // namespace elman
