#pragma once
#include "kernel.cuh"
#include <cstring>

namespace elman {
enum class FloatPointwise { Add, Subtract, Multiply };
template<FloatPointwise Op, bool Scalar>
__global__ void float_pointwise_vectors(float4* output, const float4* a, const float4* b, int count, float value) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(count)) return;
  float4 left = a[i], right;
  if constexpr (Scalar) right = make_float4(value, value, value, value);
  else right = b[i];
  if constexpr (Op == FloatPointwise::Add)
    left = make_float4(left.x + right.x, left.y + right.y, left.z + right.z, left.w + right.w);
  else if constexpr (Op == FloatPointwise::Subtract)
    left = make_float4(left.x - right.x, left.y - right.y, left.z - right.z, left.w - right.w);
  else left = make_float4(left.x * right.x, left.y * right.y, left.z * right.z, left.w * right.w);
  output[i] = left;
}
template<FloatPointwise Op>
__global__ void float_pointwise_kernel(float* output, const float* a, const float* b, int count, bool scalar, float value) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const float right = scalar ? value : b[i];
  if constexpr (Op == FloatPointwise::Add) output[i] = a[i] + right;
  else if constexpr (Op == FloatPointwise::Subtract) output[i] = a[i] - right;
  else output[i] = a[i] * right;
}
template<FloatPointwise Op>
bool try_float_pointwise(uint32_t* output, const uint32_t* a, const uint32_t* b, const KernelParams& p) {
  if (p.output_dtype() != int(DType::Float32) || p.input_dtype(KernelInput::A) != int(DType::Float32)) return false;
  int count_a = 1, count_b = 1;
  for (int d = 0; d < p.input_rank(KernelInput::A); ++d) count_a *= p.input_shape(KernelInput::A, d);
  if (count_a != p.output_count()) return false;
  // Private wrapped-scalar attribute: pass the value as a kernel argument, with no host-to-device transfer.
  const bool scalar = p.attribute(0) == 1 && p.input_rank(KernelInput::B) < 0;
  float value = 0;
  if (scalar) { const int bits = p.attribute(1); std::memcpy(&value, &bits, sizeof(value)); }
  else {
    if (p.input_dtype(KernelInput::B) != int(DType::Float32)) return false;
    for (int d = 0; d < p.input_rank(KernelInput::B); ++d) count_b *= p.input_shape(KernelInput::B, d);
    if (count_b != p.output_count()) return false;
  }
  if (p.output_count() >= 4096 && p.output_count() % 4 == 0 && reinterpret_cast<uintptr_t>(output) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(a) % 16 == 0 && (scalar || reinterpret_cast<uintptr_t>(b) % 16 == 0)) {
    const dim3 grid((unsigned(p.output_count() / 4) + 255u) / 256u);
    if (scalar) float_pointwise_vectors<Op, true><<<grid, 256, 0, current_cuda_stream()>>>(
        reinterpret_cast<float4*>(output), reinterpret_cast<const float4*>(a), nullptr, p.output_count() / 4, value);
    else float_pointwise_vectors<Op, false><<<grid, 256, 0, current_cuda_stream()>>>(
        reinterpret_cast<float4*>(output), reinterpret_cast<const float4*>(a), reinterpret_cast<const float4*>(b), p.output_count() / 4, value);
    return true;
  }
  if (p.output_count()) float_pointwise_kernel<Op><<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
      reinterpret_cast<float*>(output), reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b), p.output_count(), scalar, value);
  return true;
}
} // namespace elman
