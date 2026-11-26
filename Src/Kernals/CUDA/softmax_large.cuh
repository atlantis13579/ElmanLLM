#pragma once

namespace elman {
template <bool Maximum> __device__ float softmax_block_reduce(float value, float* shared) {
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int offset = 16; offset; offset /= 2) {
    float other = __shfl_down_sync(0xffffffff, value, offset);
    value = Maximum ? EL_MAX(value, other) : value + other;
  }
  __syncthreads();
  if (lane == 0) shared[warp] = value;
  __syncthreads();
  value = threadIdx.x < 32 ? shared[lane] : (Maximum ? -3.402823466e+38f : 0.0f);
  if (warp == 0) {
    for (int offset = 16; offset; offset /= 2) {
      float other = __shfl_down_sync(0xffffffff, value, offset);
      value = Maximum ? EL_MAX(value, other) : value + other;
    }
    if (lane == 0) shared[0] = value;
  }
  __syncthreads();
  return shared[0];
}

template <int Width, class Input> __global__ void softmax_large_rows(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                                       KernelParams p, int length, int path, Input read) {
  __shared__ float shared[32];
  int base = int(blockIdx.x) * length, thread = int(threadIdx.x);
  float maximum = -3.402823466e+38f;
  for (int j = thread; j < length; j += 1024) maximum = EL_MAX(maximum, read(a, b, p, base + j));
  maximum = softmax_block_reduce<true>(maximum, shared);
  float total = 0;
  if (path == 0) {
    // Small register rows accumulate elements spaced by one block.
    for (int j = thread; j < length; j += 1024) total += expf(read(a, b, p, base + j) - maximum);
  } else if (path == 1) {
    // Aligned rows that fit shared memory use adjacent vector elements.
    for (int start = thread * Width; start < length; start += 1024 * Width)
      for (int j = 0; j < Width; ++j) total += expf(read(a, b, p, base + start + j) - maximum);
  } else {
    // General vector traversal: align the first block, then full vector blocks
    // followed by a scalar tail. This also handles unaligned consecutive rows.
    int shift = base % Width, start = base, size = length;
    if (shift) {
      start -= shift;
      size += shift;
      if (thread >= shift && thread < size) total += expf(read(a, b, p, start + thread) - maximum);
      size -= min(1024, size);
      start += 1024;
    }
    int body = size - size % (Width * 1024);
    for (int offset = thread * Width; offset < body; offset += 1024 * Width)
      for (int j = 0; j < Width; ++j) total += expf(read(a, b, p, start + offset + j) - maximum);
    for (int offset = body + thread; offset < size; offset += 1024)
      total += expf(read(a, b, p, start + offset) - maximum);
  }
  total = softmax_block_reduce<false>(total, shared);
  for (int j = thread; j < length; j += 1024)
    WRITE_OUTPUT(base + j, expf(read(a, b, p, base + j) - maximum) / total);
}
} // namespace elman
