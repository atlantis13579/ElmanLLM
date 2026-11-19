#pragma once
#include "kernel.cuh"

namespace elman {
enum class Comparison { Equal, NotEqual, Less, LessEqual, Greater, GreaterEqual };

template<Comparison op, class T>
__device__ bool compare_values(T a, T b) {
  if constexpr (op == Comparison::Equal) return a == b;
  if constexpr (op == Comparison::NotEqual) return a != b;
  if constexpr (op == Comparison::Less) return a < b;
  if constexpr (op == Comparison::LessEqual) return a <= b;
  if constexpr (op == Comparison::Greater) return a > b;
  return a >= b;
}
// A contiguous suffix repeats at a single period. Interior broadcasts still
// need the general strided index calculation.
inline int comparison_period(const KernelParams& p, KernelInput input) {
  const int rank = p.input_rank(input);
  const int offset = p.output_rank() - rank;
  if (rank < 0 || offset < 0) return 0;
  int period = 1;
  bool repeating = false;
  for (int axis = rank - 1; axis >= 0; --axis) {
    const int extent = p.input_shape(input, axis);
    if (extent == 1) {
      if (p.output_shape(axis + offset) != 1) repeating = true;
    } else {
      if (repeating || extent != p.output_shape(axis + offset) || p.input_stride(input, axis) != period) return 0;
      period *= extent;
    }
  }
  return period;
}

template<Comparison op>
__global__ void compare_float_cuda(uint8_t* out, const float* a, const float* b,
                                   unsigned count, unsigned left, unsigned right) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) out[i] = compare_values<op>(a[i < left ? i : i % left], b[i < right ? i : i % right]);
}

template<Comparison op>
__global__ void compare_strided_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(p.output_count())) return;
  const int left = broadcast_index(int(i), KernelInput::A, p);
  const int right = broadcast_index(int(i), KernelInput::B, p);
  if (kernel_integer(p.input_dtype(KernelInput::A)) && kernel_integer(p.input_dtype(KernelInput::B))) {
    WRITE_INT_OUTPUT(i, compare_values<op>(READ_INT_A(left), READ_INT_B(right)));
  } else {
    WRITE_OUTPUT(i, float(compare_values<op>(READ_A(left), READ_B(right))));
  }
}

template<Comparison op>
void launch_comparison(uint32_t* out, const uint32_t* a, const uint32_t* b, const KernelParams& p) {
  const unsigned count = unsigned(p.output_count());
  if (!count) return;
  const unsigned blocks = (count + 255) / 256;
  auto stream = current_cuda_stream();
  if (p.input_dtype(KernelInput::A) == int(DType::Float32) && p.input_dtype(KernelInput::B) == int(DType::Float32) &&
      p.output_dtype() == int(DType::Bool)) {
    const int left = comparison_period(p, KernelInput::A), right = comparison_period(p, KernelInput::B);
    if (left && right) {
      compare_float_cuda<op><<<blocks, 256, 0, stream>>>(reinterpret_cast<uint8_t*>(out),
          reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b), count, left, right);
      return;
    }
  }
  compare_strided_cuda<op><<<blocks, 256, 0, stream>>>(out, a, b, p);
}
} // namespace elman
