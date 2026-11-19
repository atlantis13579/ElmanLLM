#include "kernel.cuh"
#include "short_row_reduction.cuh"

namespace elman {
__device__ void rms_norm_write(uint32_t* out, const uint32_t* a, const uint32_t* b,
                               const uint32_t* c, const KernelParams& p, int i, int j, float inverse) {
  float value = __fmul_rn(READ_A(i), inverse);
  if (p.attribute(operator_attr::normalization::kRounding))
    value = elman_cuda_decode(elman_cuda_encode(value, p.output_dtype()), p.output_dtype());
  if (p.input_rank(KernelInput::B) >= 0) value = __fmul_rn(value, READ_B(j));
  if (p.input_rank(KernelInput::C) >= 0) value = __fadd_rn(value, READ_C(j));
  WRITE_OUTPUT(i, value);
}
__global__ void rms_norm_short_rows_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                        const uint32_t* c, KernelParams p, int length, int rows, float factor) {
  __shared__ float inverses[512];
  unsigned row = blockIdx.x * blockDim.y + threadIdx.y;
  float total = short_row_sum<true>(a, p, row, rows, length);
  if (threadIdx.x == 0)
    inverses[threadIdx.y] = rsqrtf(__fadd_rn(__fmul_rn(total, factor), float_attribute(p, operator_attr::normalization::kEpsilon)));
  __syncthreads();
  if (row < unsigned(rows))
    for (int j = int(threadIdx.x); j < length; j += int(blockDim.x))
      rms_norm_write(out, a, b, c, p, row * length + j, j, inverses[threadIdx.y]);
}
// Match mean_rows_cuda after the eager Float32 square, including the row-count
// dependent tree and separate multiply/add rounding boundaries.
__global__ void rms_norm_rows_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, KernelParams p,
                               int length, int rows, bool shared_row, float factor) {
  __shared__ float partial[512], inverses[512];
  int lane = int(threadIdx.x), warp = int(threadIdx.y), width = int(blockDim.x);
  int row = shared_row ? int(blockIdx.x) : int(blockIdx.x * blockDim.y + threadIdx.y);
  int first = shared_row ? lane + warp * width : lane;
  int step = shared_row ? width * int(blockDim.y) : width;
  float accum[4] = {0, 0, 0, 0};
  if (row < rows) {
    for (int group = first; group < length / 4; group += step) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        float x = READ_A(row * length + group * 4 + j);
        accum[j] = __fadd_rn(accum[j], __fmul_rn(x, x));
      }
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
  if (lane == 0 && (!shared_row || warp == 0))
    inverses[shared_row ? 0 : warp] = rsqrtf(__fadd_rn(__fmul_rn(value, factor), float_attribute(p, operator_attr::normalization::kEpsilon)));
  __syncthreads();
  if (row < rows) {
    float inverse = inverses[shared_row ? 0 : warp];
    for (int j = first; j < length; j += step) rms_norm_write(out, a, b, c, p, row * length + j, j, inverse);
  }
}
// The eager mean uses a sequential sum outside its cooperative shape range.
// Compute that sum once per row and distribute the output across the block.
__global__ void rms_norm_sequential_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                        const uint32_t* c, KernelParams p, int length) {
  __shared__ float inverse;
  int base = int(blockIdx.x) * length;
  if (threadIdx.x == 0) {
    float total = 0;
    for (int j = 0; j < length; ++j) {
      float x = READ_A(base + j);
      total = __fadd_rn(total, __fmul_rn(x, x));
    }
    inverse = rsqrtf(__fadd_rn(total / float(length), float_attribute(p, operator_attr::normalization::kEpsilon)));
  }
  __syncthreads();
  for (int j = int(threadIdx.x); j < length; j += int(blockDim.x)) rms_norm_write(out, a, b, c, p, base + j, j, inverse);
}
void launch_rms_norm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const uint32_t* c, const KernelParams& p) {
  if (!p.output_count()) return;
  int length = 1;
  for (int axis = p.output_rank() - p.attribute(operator_attr::normalization::kTrailingDimensions);
       axis < p.output_rank(); ++axis) length *= p.output_shape(axis);
  int rows = p.output_count() / length;
  if (length <= 128) {
    auto block = short_row_block(length, rows);
    rms_norm_short_rows_cuda<<<(rows + block.y - 1) / block.y, block, 0, current_cuda_stream()>>>(
        out, a, b, c, p, length, rows, 1.0f / float(length));
    return;
  }
  if (length > 128 && length <= 8192 && length % 4 == 0) {
    auto power = [](int value) { int result = 1; while (result <= value / 2 && result < 512) result *= 2; return result; };
    int columns = power(length / 4), row_power = power(rows);
    int width = columns < 32 ? columns : 32;
    int height = row_power < 512 / width ? row_power : 512 / width;
    width = columns < 512 / height ? columns : 512 / height;
    int threshold = height * 16 < 256 ? height * 16 : 256;
    bool shared_row = (length + width - 1) / width >= threshold;
    int grid = shared_row ? rows : (rows + height - 1) / height;
    rms_norm_rows_cuda<<<grid, dim3(width, height), 0, current_cuda_stream()>>>(out, a, b, c, p, length, rows, shared_row, 1.0f / float(length));
  } else rms_norm_sequential_cuda<<<rows, 256, 0, current_cuda_stream()>>>(out, a, b, c, p, length);
}
} // namespace elman
