#pragma once

#include "kernel.cuh"

namespace elman {
enum class RowReduction { Sum, Product, Minimum, Maximum };

template <RowReduction operation>
__global__ void reduce_contiguous_rows(uint32_t* out, const uint32_t* a, int length, KernelParams p) {
  constexpr int threads = 128;
  constexpr bool sum = operation == RowReduction::Sum;
  constexpr bool product = operation == RowReduction::Product;
  constexpr bool minimum = operation == RowReduction::Minimum;
  const int base = int(blockIdx.x) * length;
  __shared__ float values[threads];
  [[maybe_unused]] __shared__ int positions[threads];

  [[maybe_unused]] auto choose = [](float current, int current_index, float candidate, int candidate_index) {
    if (isnan(current) || isnan(candidate))
      return isnan(candidate) && (!isnan(current) || candidate_index < current_index);
    return (minimum ? candidate < current : candidate > current) ||
           (candidate == current && candidate_index > current_index);
  };
  float value = sum ? 0.f : product ? 1.f : minimum ? EL_INF : -EL_INF;
  [[maybe_unused]] int position = -1;
  for (int i = threadIdx.x; i < length; i += threads) {
    const float x = READ_A(base + i);
    if constexpr (sum) value += x;
    else if constexpr (product) value *= x;
    else if (choose(value, position, x, i)) { value = x; position = i; }
  }
  values[threadIdx.x] = value;
  if constexpr (!sum && !product) positions[threadIdx.x] = position;
  __syncthreads();
  for (int step = threads / 2; step; step /= 2) {
    if (threadIdx.x < step) {
      const float other = values[threadIdx.x + step];
      if constexpr (sum) value += other;
      else if constexpr (product) value *= other;
      else {
        const int other_position = positions[threadIdx.x + step];
        if (choose(value, position, other, other_position)) { value = other; position = other_position; }
        positions[threadIdx.x] = position;
      }
      values[threadIdx.x] = value;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) WRITE_OUTPUT(int(blockIdx.x), value);
}

template <RowReduction operation>
bool try_reduce_contiguous_rows(uint32_t* out, const uint32_t* a, const KernelParams& p) {
  const auto dtype = DType(p.output_dtype());
  if (!is_compute_float(dtype) || p.input_dtype(KernelInput::A) != p.output_dtype()) return false;
  if constexpr (operation == RowReduction::Sum || operation == RowReduction::Product) {
    if (dtype != DType::Float32) return false;
  }
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  if (axis >= 0 && p.input_stride(KernelInput::A, axis) != 1) return false;
  int length = axis < 0 ? 1 : p.input_shape(KernelInput::A, axis);
  if (axis < 0)
    for (int dim = 0; dim < p.input_rank(KernelInput::A); ++dim) length *= p.input_shape(KernelInput::A, dim);
  if (length < 256) return false;
  reduce_contiguous_rows<operation><<<p.output_count(), 128, 0, current_cuda_stream()>>>(out, a, length, p);
  return true;
}
}  // namespace elman
