#include "kernel.cuh"
#include "row_reduce.cuh"

namespace elman {
namespace {
constexpr int threads = 128;
__global__ void layer_norm_rows(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                const uint32_t* c, int length, KernelParams p) {
  __shared__ float scratch[threads];
  const int base = int(blockIdx.x) * length;
  float sum = 0;
  for (int j = threadIdx.x; j < length; j += threads) sum += READ_A(base + j);
  const float mean = row_reduce<false>(sum, scratch) / float(length);
  float squares = 0;
  for (int j = threadIdx.x; j < length; j += threads) {
    const float delta = READ_A(base + j) - mean;
    squares += __fmul_rn(delta, delta);
  }
  const float variance = row_reduce<false>(squares, scratch) / float(length);
  const float inverse = rsqrtf(variance + float_attribute(p, kernal_attr::normalization::kEpsilon));
  for (int j = threadIdx.x; j < length; j += threads) {
    const int i = base + j;
    float value = (READ_A(i) - mean) * inverse;
    if (b) value = __fmul_rn(value, READ_B(j));
    if (c) value = __fadd_rn(value, READ_C(j));
    WRITE_OUTPUT(i, value);
  }
}
} // namespace
__global__ void kernel_layer_norm_cuda(uint32_t* out, const uint32_t* a,
                                       const uint32_t* b, const uint32_t* c,
                                       KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int length = 1;
    for (int axis =
             p.input_rank(KernelInput::A) -
             p.attribute(kernal_attr::normalization::kTrailingDimensions);
         axis < p.input_rank(KernelInput::A); ++axis) {
      length *= p.input_shape(KernelInput::A, axis);
    }
    int base = (i / length) * length;
    float average = 0.0f;
    for (int j = 0; j < length; ++j) {
      average += READ_A(base + j);
    }
    average /= float(length);
    float variance = 0.0f;
    for (int j = 0; j < length; ++j) {
      float difference = READ_A(base + j) - average;
      variance += difference * difference;
    }
    variance /= float(length);
    float value =
        (READ_A(i) - average) /
        EL_SQRT(variance +
                float_attribute(p, kernal_attr::normalization::kEpsilon));
    int coordinate = i % length;
    if (p.input_rank(KernelInput::B) >= 0) {
      value *= READ_B(coordinate);
    }
    if (p.input_rank(KernelInput::C) >= 0) {
      value += READ_C(coordinate);
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_layer_norm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    const int rank = p.input_rank(KernelInput::A);
    int length = 1;
    for (int axis = rank - p.attribute(kernal_attr::normalization::kTrailingDimensions); axis < rank; ++axis)
      length *= p.input_shape(KernelInput::A, axis);
    if (length >= 128 && p.output_dtype() == int(DType::Float32)) {
      layer_norm_rows<<<p.output_count() / length, threads, 0, current_cuda_stream()>>>(out, a, b, c, length, p);
      return;
    }
    kernel_layer_norm_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
