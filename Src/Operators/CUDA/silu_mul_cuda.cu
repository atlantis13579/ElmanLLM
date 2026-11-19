#include "kernel.cuh"

namespace elman {
namespace {
template<int DTypeCode>
__global__ void contiguous_silu_mul_cuda(uint32_t* out, const uint32_t* gate, const uint32_t* up,
                                         unsigned count) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const float x = elman_cuda_decode(elman_cuda_load(gate, i, DTypeCode), DTypeCode);
  const float y = elman_cuda_decode(elman_cuda_load(up, i, DTypeCode), DTypeCode);
  const float activation = elman_cuda_decode(elman_cuda_encode(x / (1.0f + EL_EXP(-x)), DTypeCode), DTypeCode);
  elman_cuda_store(out, i, elman_cuda_encode(activation * y, DTypeCode), DTypeCode);
}

bool matches_output(const KernelParams& p, KernelInput input) {
  const int rank = p.output_rank();
  if (p.input_rank(input) != rank || p.input_dtype(input) != p.output_dtype()) return false;
  int stride = 1;
  for (int axis = rank - 1; axis >= 0; --axis) {
    if (p.input_shape(input, axis) != p.output_shape(axis) || p.input_stride(input, axis) != stride) return false;
    stride *= p.output_shape(axis);
  }
  return true;
}
} // namespace

__global__ void kernel_silu_mul_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, KernelParams p) {
  unsigned element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element >= unsigned(p.output_count())) return;
  int i = int(element), dtype = p.output_dtype();
  float x = READ_A(broadcast_index(i, KernelInput::A, p));
  float activation = x / (1.0f + EL_EXP(-x));
  activation = elman_cuda_decode(elman_cuda_encode(activation, dtype), dtype);
  WRITE_OUTPUT(i, activation * READ_B(broadcast_index(i, KernelInput::B, p)));
}
void launch_silu_mul_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const uint32_t*, const KernelParams& p) {
  const unsigned count = p.output_count();
  if (!count) return;
  const unsigned blocks = (count + 255u) / 256u;
  if (matches_output(p, KernelInput::A) && matches_output(p, KernelInput::B)) {
    switch (DType(p.output_dtype())) {
      case DType::Float32:
        contiguous_silu_mul_cuda<int(DType::Float32)><<<blocks, 256, 0, current_cuda_stream()>>>(out, a, b, count);
        return;
      case DType::Float16:
        contiguous_silu_mul_cuda<int(DType::Float16)><<<blocks, 256, 0, current_cuda_stream()>>>(out, a, b, count);
        return;
      case DType::BFloat16:
        contiguous_silu_mul_cuda<int(DType::BFloat16)><<<blocks, 256, 0, current_cuda_stream()>>>(out, a, b, count);
        return;
      default:
        break;
    }
  }
  kernel_silu_mul_cuda<<<blocks, 256, 0, current_cuda_stream()>>>(out, a, b, p);
}
} // namespace elman
