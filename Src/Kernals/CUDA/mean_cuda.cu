#include "kernel.cuh"
#include "short_row_reduction.cuh"

namespace elman {
__global__ void mean_short_rows_cuda(uint32_t* out, const uint32_t* a, KernelParams p, int length, float factor) {
  unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
  float total = short_row_sum<false>(a, p, row, p.output_count(), length);
  if (row < unsigned(p.output_count()) && threadIdx.x == 0) WRITE_OUTPUT(row, __fmul_rn(total, factor));
}
// Contiguous rows use four independent Float32 accumulators and an ordered
// tree before the result is rounded to the output dtype.
__global__ void mean_rows_cuda(uint32_t* out, const uint32_t* a, KernelParams p,
                               int length, bool shared_row, float factor) {
  __shared__ float partial[512];
  int lane = int(threadIdx.x), warp = int(threadIdx.y), width = int(blockDim.x);
  int row = shared_row ? int(blockIdx.x) : int(blockIdx.x * blockDim.y + threadIdx.y);
  int first = shared_row ? lane + warp * width : lane;
  int step = shared_row ? width * int(blockDim.y) : width;
  float accum[4] = {0, 0, 0, 0};
  if (row < p.output_count()) {
    for (int group = first; group < length / 4; group += step) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) accum[j] += READ_A(row * length + group * 4 + j);
    }
  }
  float value = ((accum[0] + accum[1]) + accum[2]) + accum[3];
  int address = warp * width + lane;
  if (shared_row) {
    partial[address] = value;
    for (int offset = int(blockDim.y) / 2; offset > 0; offset >>= 1) {
      __syncthreads();
      if (warp < offset) {
        value += partial[address + offset * width];
        partial[address] = value;
      }
    }
    __syncthreads();
  }
  if (width > 32) {
    partial[address] = value;
    for (int offset = width / 2; offset >= 32; offset >>= 1) {
      __syncthreads();
      if (lane < offset) {
        value += partial[address + offset];
        partial[address] = value;
      }
    }
  }
  __syncthreads();
  for (int offset = 1; offset < 32; offset <<= 1) value += __shfl_down_sync(0xffffffff, value, offset);
  if (row < p.output_count() && lane == 0 && (!shared_row || warp == 0)) WRITE_OUTPUT(row, value * factor);
}
__global__ void kernel_mean_cuda(uint32_t* out, const uint32_t* a,
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
    float value = 0.0f;
    for (int j = 0; j < length; ++j) {
      float x = READ_A(base + j * inner);
      value = value + x;
    }
    WRITE_OUTPUT(i, value / float(length));
  }
}
void launch_mean_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    int axis = p.attribute(kernal_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = axis < 0 ? 1 : p.input_shape(KernelInput::A, axis);
    if (axis < 0) for (int i = 0; i < p.input_rank(KernelInput::A); ++i) length *= p.input_shape(KernelInput::A, i);
    if (inner == 1 && length > 0 && length <= 128 &&
        p.input_dtype(KernelInput::A) == int(DType::Float32) && p.output_dtype() == int(DType::Float32)) {
      dim3 block = short_row_block(length, p.output_count());
      mean_short_rows_cuda<<<(p.output_count() + block.y - 1) / block.y, block, 0, current_cuda_stream()>>>(
          out, a, p, length, 1.0f / float(length));
      return;
    }
    if (inner == 1 && length > 128 && length <= 8192 && length % 4 == 0 &&
        is_compute_float(DType(p.output_dtype())) && p.input_dtype(KernelInput::A) == p.output_dtype()) {
      int (*const power)(int) = [](int value) { int result = 1; while (result <= value / 2 && result < 512) result *= 2; return result; };
      int columns = power(length / 4), rows = power(p.output_count());
      int width = columns < 32 ? columns : 32;
      int height = rows < 512 / width ? rows : 512 / width;
      width = columns < 512 / height ? columns : 512 / height;
      int threshold = height * 16 < 256 ? height * 16 : 256;
      bool shared_row = (length + width - 1) / width >= threshold;
      int grid = shared_row ? p.output_count() : (p.output_count() + height - 1) / height;
      mean_rows_cuda<<<grid, dim3(width, height), 0, current_cuda_stream()>>>(out, a, p, length, shared_row, 1.0f / float(length));
      return;
    }
    kernel_mean_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
