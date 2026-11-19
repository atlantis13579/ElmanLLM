#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace elman {
template<class ReadLeft, class ReadRight>
float contiguous_float_dot(int count, ReadLeft left, ReadRight right) {
  // The native non-BLAS transposed GEMM uses four independent accumulators.
  float partial[4]{};
  int i = 0;
  for (; i <= count - 4; i += 4)
    for (int lane = 0; lane < 4; ++lane) partial[lane] += left(i + lane) * right(i + lane);
  for (; i < count; ++i) partial[0] += left(i) * right(i);
  for (int lane = 1; lane < 4; ++lane) partial[0] += partial[lane];
  return partial[0];
}

template<class ReadLeft, class ReadRight>
float contiguous_narrow_dot(int count, ReadLeft left, ReadRight right) {
  // Preserve the AVX2 reference's eight independent FP32 vector accumulators,
  // pairwise register reduction, vector tail and final scalar tail. A serial
  // dot can land on the opposite side of a narrow rounding midpoint.
  float partial[8][8]{};
  int offset = 0;
  for (; offset <= count - 64; offset += 64)
    for (int group = 0; group < 8; ++group)
      for (int lane = 0; lane < 8; ++lane) {
        int index = offset + group * 8 + lane;
        partial[group][lane] = std::fma(left(index), right(index), partial[group][lane]);
      }
  for (int distance = 4; distance; distance /= 2)
    for (int group = 0; group < distance; ++group)
      for (int lane = 0; lane < 8; ++lane)
        partial[group][lane] += partial[group + distance][lane];
  auto reduce = [](float* lanes) {
#if defined(__GNUC__) && __GNUC__ > 5 && !defined(_MSC_VER)
    for (int distance = 4; distance; distance /= 2)
      for (int lane = 0; lane < distance; ++lane) lanes[lane] += lanes[lane + distance];
    return lanes[0];
#else
    float total = lanes[0];
    for (int lane = 1; lane < 8; ++lane) total += lanes[lane];
    return total;
#endif
  };
  float total = reduce(partial[0]), tail[8]{};
  for (; offset <= count - 16; offset += 16)
    for (int half = 0; half < 2; ++half)
      for (int lane = 0; lane < 8; ++lane) {
        int index = offset + half * 8 + lane;
        tail[lane] = std::fma(left(index), right(index), tail[lane]);
      }
  total += reduce(tail);
  for (; offset < count; ++offset) total += left(offset) * right(offset);
  return total;
}

template<class Read>
float contiguous_float_sum(int count, Read read) {
  // Four interleaved eight-lane groups with hierarchical accumulation match
  // the AVX2 reference's contiguous Float32 sum without requiring SIMD here.
  const int vectors = count / 8, groups = vectors / 4;
  int logarithm = 0;
  for (int value = groups - 1; value > 0; value >>= 1) ++logarithm;
  const int power = std::max(4, logarithm / 4), block = 1 << power;
  float sums[4][4][8] = {};
  int group = 0;
  auto accumulate = [&] {
    for (int part = 0; part < 4; ++part)
      for (int lane = 0; lane < 8; ++lane)
        sums[0][part][lane] += read(group * 32 + part * 8 + lane);
    ++group;
  };
  while (group + block <= groups) {
    for (int i = 0; i < block; ++i) accumulate();
    for (int level = 1; level < 4; ++level) {
      for (int part = 0; part < 4; ++part)
        for (int lane = 0; lane < 8; ++lane) {
          sums[level][part][lane] += sums[level - 1][part][lane];
          sums[level - 1][part][lane] = 0;
        }
      if (int64_t(group) & (int64_t(block - 1) << (level * power))) break;
    }
  }
  while (group < groups) accumulate();
  for (int level = 1; level < 4; ++level)
    for (int part = 0; part < 4; ++part)
      for (int lane = 0; lane < 8; ++lane) sums[0][part][lane] += sums[level][part][lane];
  for (int vector = groups * 4; vector < vectors; ++vector)
    for (int lane = 0; lane < 8; ++lane) sums[0][0][lane] += read(vector * 8 + lane);
  for (int part = 1; part < 4; ++part)
    for (int lane = 0; lane < 8; ++lane) sums[0][0][lane] += sums[0][part][lane];
  float total = 0;
  for (int i = vectors * 8; i < count; ++i) total += read(i);
  for (int lane = 0; lane < 8; ++lane) total += sums[0][0][lane];
  return total;
}
} // namespace elman
