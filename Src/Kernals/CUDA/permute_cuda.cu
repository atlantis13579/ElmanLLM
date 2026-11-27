#include "kernel.cuh"

namespace elman {
__global__ void permute_head_vectors(uint4* output, const uint4* input, int count, int first, int second, int width) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(count)) return;
  const int row = int(i) / width;
  const int outer = row / (first * second), a = row % first, b = (row / first) % second;
  output[i] = input[((outer * first + a) * second + b) * width + int(i) % width];
}
__global__ void transpose_float_tiles(uint32_t* output, const uint32_t* input, int rows, int columns) {
  __shared__ uint32_t tile[32][33];
  const int x = int(blockIdx.x) * 32 + int(threadIdx.x), y = int(blockIdx.y) * 32 + int(threadIdx.y);
  const size_t base = size_t(blockIdx.z) * rows * columns;
  for (int j = 0; j < 32; j += 8)
    if (x < columns && y + j < rows) tile[threadIdx.y + j][threadIdx.x] = input[base + size_t(y + j) * columns + x];
  __syncthreads();
  const int out_x = int(blockIdx.y) * 32 + int(threadIdx.x), out_y = int(blockIdx.x) * 32 + int(threadIdx.y);
  for (int j = 0; j < 32; j += 8)
    if (out_x < rows && out_y + j < columns) output[base + size_t(out_y + j) * rows + out_x] = tile[threadIdx.x][threadIdx.y + j];
}
__global__ void kernel_permute_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int flat = i;
    int source = 0;
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      source += coordinate *
                p.input_stride(
                    KernelInput::A,
                    p.attribute(kernal_attr::permutation::kFirstAxis + axis));
    }
    COPY_INPUT_A(i, source);
  }
}
void launch_permute_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    const int rank = p.output_rank();
    if (rank == 4 && p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == p.output_dtype() &&
        p.attribute(0) == 0 && p.attribute(1) == 2 && p.attribute(2) == 1 && p.attribute(3) == 3 &&
        p.output_shape(3) % 4 == 0 && reinterpret_cast<uintptr_t>(out) % 16 == 0 && reinterpret_cast<uintptr_t>(a) % 16 == 0) {
      permute_head_vectors<<<(unsigned(p.output_count() / 4) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
          reinterpret_cast<uint4*>(out), reinterpret_cast<const uint4*>(a), p.output_count() / 4,
          p.input_shape(KernelInput::A, 1), p.input_shape(KernelInput::A, 2), p.output_shape(3) / 4);
      return;
    }
    if (rank >= 2 && p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == p.output_dtype()) {
      bool transpose = p.attribute(rank - 2) == rank - 1 && p.attribute(rank - 1) == rank - 2;
      for (int d = 0; d < rank - 2; ++d) transpose = transpose && p.attribute(d) == d;
      const int rows = p.input_shape(KernelInput::A, rank - 2), columns = p.input_shape(KernelInput::A, rank - 1);
      const int batches = p.output_count() / (rows * columns);
      if (transpose && batches <= 65535 && (unsigned(rows) + 31u) / 32u <= 65535) {
        transpose_float_tiles<<<dim3((unsigned(columns) + 31u) / 32u, (unsigned(rows) + 31u) / 32u, batches), dim3(32, 8), 0, current_cuda_stream()>>>(out, a, rows, columns);
        return;
      }
    }
    kernel_permute_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
