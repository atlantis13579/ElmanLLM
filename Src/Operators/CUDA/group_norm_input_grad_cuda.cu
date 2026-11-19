#include "kernel.cuh"
#include "row_reduce.cuh"

namespace elman {
namespace {
constexpr int threads = 256;
__global__ void group_norm_input_grad_rows(uint32_t* out, const uint32_t* a,
                                          const uint32_t* b, const uint32_t* c,
                                          int channels, int spatial, int length, KernelParams p) {
  __shared__ float scratch[threads];
  const int base = int(blockIdx.x) * length;
  const float reciprocal = 1.f / float(length);
  float sum = 0;
  for (int j = threadIdx.x; j < length; j += threads) sum += READ_A(base + j);
  const float mean = row_reduce<false>(sum, scratch) * reciprocal;
  float squares = 0, weighted = 0, cross = 0;
  for (int j = threadIdx.x; j < length; j += threads) {
    const int i = base + j;
    const float delta = READ_A(i) - mean;
    const float dy = READ_C(i) * READ_B((i / spatial) % channels);
    squares = fmaf(delta, delta, squares);
    cross = fmaf(dy, delta, cross);
    weighted += dy;
  }
  const float variance = row_reduce<false>(squares, scratch) * reciprocal;
  const float average_dy = row_reduce<false>(weighted, scratch) * reciprocal;
  const float inverse = rsqrtf(variance + float_attribute(p, operator_attr::group_norm::kEpsilon));
  const float slope = row_reduce<false>(cross, scratch) * reciprocal * inverse * inverse;
  for (int j = threadIdx.x; j < length; j += threads) {
    const int i = base + j;
    const float dy = READ_C(i) * READ_B((i / spatial) % channels);
    WRITE_OUTPUT(i, (dy - average_dy - (READ_A(i) - mean) * slope) * inverse);
  }
}
} // namespace
__global__ void kernel_group_norm_input_grad_cuda(uint32_t* out, const uint32_t* a,
                                                  const uint32_t* b, const uint32_t* c,
                                                  KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
    int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
    int per_group = channels / p.attribute(operator_attr::group_norm::kGroups);
    int length = per_group * spatial;
    int base = (i / length) * length;
    float average = 0;
    for (int j = 0; j < length; ++j) average += READ_A(base + j);
    average /= float(length);
    float variance = 0;
    for (int j = 0; j < length; ++j) {
      float difference = READ_A(base + j) - average;
      variance += difference * difference;
    }
    variance /= float(length);
    float epsilon = float_attribute(p, operator_attr::group_norm::kEpsilon);
    const int dtype = p.input_dtype(KernelInput::A);
    if (dtype == 3 || dtype == 4) epsilon = elman_cuda_decode(elman_cuda_encode(epsilon, dtype), dtype);
    float inverse = rsqrtf(variance + epsilon);
    if (dtype == 3 || dtype == 4) {
      average = elman_cuda_decode(elman_cuda_encode(average, dtype), dtype);
      inverse = elman_cuda_decode(elman_cuda_encode(inverse, dtype), dtype);
    }
    float ds = 0, db = 0;
    for (int channel = 0; channel < per_group; ++channel) {
      float channel_ds = 0, channel_db = 0;
      for (int j = 0; j < spatial; ++j) {
        int index = base + channel * spatial + j;
        float product = READ_C(index) * READ_A(index);
        // Round the product to the input dtype when the spatial extent is one.
        if (spatial == 1 && (dtype == 3 || dtype == 4))
          product = elman_cuda_decode(elman_cuda_encode(product, dtype), dtype);
        channel_ds += product;
        channel_db += READ_C(index);
      }
      float weight = READ_B((base / spatial + channel) % channels);
      ds += channel_ds * weight;
      db += channel_db * weight;
    }
    const float inverse_length = 1.0f / float(length);
    float coefficient2 = fmaf(db, average, -ds) * inverse * inverse * inverse * inverse_length;
    // Match the reference's FMA operand order and coefficient-store boundary.
    float coefficient3 = fmaf(-coefficient2, average, -db * inverse * inverse_length);
    float coefficient1 = inverse * READ_B((i / spatial) % channels);
    float value = fmaf(coefficient1, READ_C(i), __fmul_rn(coefficient2, READ_A(i)));
    WRITE_OUTPUT(i, __fadd_rn(value, coefficient3));
  }
}
void launch_group_norm_input_grad_cuda(uint32_t* out, const uint32_t* a,
                                      const uint32_t* b, const uint32_t* c,
                                      const KernelParams& p) {
  if (p.output_count() > 0) {
    const int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
    const int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
    const int length = channels / p.attribute(operator_attr::group_norm::kGroups) * spatial;
    if (length >= 128 && p.output_dtype() == int(DType::Float32)) {
      group_norm_input_grad_rows<<<p.output_count() / length, threads, 0, current_cuda_stream()>>>(
          out, a, b, c, channels, spatial, length, p);
      return;
    }
    kernel_group_norm_input_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, a, b, c, p);
  }
}
} // namespace elman
