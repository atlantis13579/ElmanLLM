#include "kernel.cuh"
#include "long_row_reduction.cuh"
#include "integer_reduction.cuh"
#include "Core/AllocationPool.h"

namespace elman {
namespace {
__global__ void float_column_sum_partial_kernel(float* partial, const float* input, int length, int inner, int parts) {
  __shared__ float sums[8][32];
  const int column = int(blockIdx.x) * 32 + int(threadIdx.x), lane = int(threadIdx.y);
  const int part = int(blockIdx.y), outer = int(blockIdx.z);
  float value = 0;
  if (column < inner) for (int j = part * 128 + lane; j < length && j < (part + 1) * 128; j += 8)
    value += input[(outer * length + j) * inner + column];
  sums[lane][threadIdx.x] = value;
  __syncthreads();
  for (int offset = 4; offset; offset /= 2) {
    if (lane < offset) sums[lane][threadIdx.x] += sums[lane + offset][threadIdx.x];
    __syncthreads();
  }
  if (!lane && column < inner) partial[(outer * parts + part) * inner + column] = sums[0][threadIdx.x];
}
__global__ void float_column_sum_finish_kernel(float* output, const float* partial, int count, int inner, int parts) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const int base = (i / inner) * parts * inner + i % inner;
  float value = 0;
  for (int part = 0; part < parts; ++part) value += partial[base + part * inner];
  output[i] = value;
}
}
__global__ void kernel_sum_cuda(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(kernal_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = axis < 0 ? input_count(KernelInput::A, p)
                          : p.input_shape(KernelInput::A, axis);
    int base = axis < 0 ? 0 : (i / inner) * length * inner + i % inner;
    if (kernel_integer(p.output_dtype())) {
      uint32_t integer_value = 0u;
      for (int j = 0; j < length; ++j) {
        integer_value += uint32_t(READ_INT_A(base + j * inner));
      }
      WRITE_INT_OUTPUT(i, int32_t(integer_value));
      return;
    }
    float value = 0.0f;
    for (int j = 0; j < length; ++j) {
      float x = READ_A(base + j * inner);
      value = value + x;
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_sum_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    if (p.output_dtype() == int(DType::Int64)) {
      integer_reduction_cuda<false><<<p.output_count(), 256, 0, current_cuda_stream()>>>(reinterpret_cast<uint64_t*>(out), a, p);
      return;
    }
    if (try_reduce_contiguous_rows<RowReduction::Sum>(out, a, p)) return;
    const int axis = p.attribute(kernal_attr::reduction::kAxis);
    if (p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == int(DType::Float32) && axis >= 0) {
      const int inner = p.input_stride(KernelInput::A, axis), length = p.input_shape(KernelInput::A, axis);
      const int parts = length / 128 + int(length % 128 != 0), outer = p.output_count() / inner;
      if (inner >= 32 && length >= 32 && outer <= 65535 && parts <= 65535) {
        BufferPtr storage = backend_workspace(Backend::CUDA, checked_memory_mul(checked_memory_mul(size_t(p.output_count()), size_t(parts)), sizeof(float)));
        float* partial = static_cast<float*>(storage->raw_pointer());
        float_column_sum_partial_kernel<<<dim3((inner + 31) / 32, parts, outer), dim3(32, 8), 0, current_cuda_stream()>>>(
            partial, reinterpret_cast<const float*>(a), length, inner, parts);
        float_column_sum_finish_kernel<<<(p.output_count() + 255) / 256, 256, 0, current_cuda_stream()>>>(
            reinterpret_cast<float*>(out), partial, p.output_count(), inner, parts);
        return;
      }
    }
    kernel_sum_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
