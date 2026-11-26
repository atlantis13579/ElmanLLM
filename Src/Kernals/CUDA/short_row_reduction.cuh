#pragma once
#include "kernel.cuh"

namespace elman {
inline dim3 short_row_block(int length, int rows) {
  int (*const power)(int) = [](int value) { int result = 1; while (result <= value / 2 && result < 512) result *= 2; return result; };
  int columns = power(length), row_power = power(rows);
  int width = columns < 32 ? columns : 32;
  int height = row_power < 512 / width ? row_power : 512 / width;
  width = columns < 512 / height ? columns : 512 / height;
  return dim3(width, height);
}

template<bool Square>
__device__ float short_row_sum(const uint32_t* a, const KernelParams& p, unsigned row, int rows, int length) {
  // Short contiguous reductions use scalar lanes, unlike the four-element
  // vector loads of longer rows. Match their row-count-dependent reduction.
  __shared__ float partial[512];
  int lane = int(threadIdx.x), width = int(blockDim.x);
  float terms[4]{};
  if (row < unsigned(rows)) {
    #pragma unroll
    for (int part = 0; part < 4; ++part) {
      int column = lane + part * width;
      if (column < length) {
        float x = READ_A(row * length + column);
        terms[part] = __fadd_rn(0.0f, Square ? __fmul_rn(x, x) : x);
      }
    }
  }
  float total = __fadd_rn(__fadd_rn(__fadd_rn(terms[0], terms[1]), terms[2]), terms[3]);
  int address = int(threadIdx.y) * width + lane;
  if (width > 32) {
    partial[address] = total;
    for (int distance = width / 2; distance >= 32; distance /= 2) {
      __syncthreads();
      if (lane < distance) {
        total = __fadd_rn(total, partial[address + distance]);
        partial[address] = total;
      }
    }
  }
  __syncthreads();
  int lanes = width < 32 ? width : 32;
  for (int distance = 1; distance < lanes; distance *= 2)
    total = __fadd_rn(total, __shfl_down_sync(__activemask(), total, distance));
  return total;
}
} // namespace elman
