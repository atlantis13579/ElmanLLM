#include "kernel.cuh"
#include "row_reduce.cuh"

namespace elman {
namespace {
__global__ void flux_rotary_pairs(uint32_t* out, const uint32_t* a, const uint32_t* b, KernelParams p) {
  const int pair = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (pair >= p.output_count() / 2) {
    return;
  }
  const int i = pair * 2, tokens = p.output_shape(2), heads = p.output_shape(1);
  const int row = i / 128, batch = p.input_shape(KernelInput::B, 0) == 1 ? 0 : row / tokens / heads;
  const int table = (batch * tokens + row % tokens) * 128 + i % 128;
  const float cs = READ_B(table), sn = READ_B(table + 1);
  const float x = READ_A(i), y = READ_A(i + 1);
  WRITE_OUTPUT(i, __fsub_rn(__fmul_rn(x, cs), __fmul_rn(y, sn)));
  WRITE_OUTPUT(i + 1, __fadd_rn(__fmul_rn(y, cs), __fmul_rn(x, sn)));
}

__device__ float round_activation(float value, int dtype) {
  return elman_cuda_decode(elman_cuda_encode(value, dtype), dtype);
}

constexpr int threads = 128;
__global__ void flux_modulate_rows(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                                   KernelParams p) {
  __shared__ float scratch[threads];
  const int width = p.output_shape(2), dtype = p.output_dtype(), start = int(blockIdx.x) * width;
  const int mod = p.input_shape(KernelInput::B, 1) == 1 ? int(blockIdx.x) / p.output_shape(1) * width : start;
  float sum = 0;
  for (int d = threadIdx.x; d < width; d += threads) {
    sum += READ_A(start + d);
  }
  const float average = row_reduce<false>(sum, scratch) / width;
  float squares = 0;
  for (int d = threadIdx.x; d < width; d += threads) {
    const float delta = READ_A(start + d) - average;
    squares += delta * delta;
  }
  const float inverse = rsqrtf(row_reduce<false>(squares, scratch) / width + 1e-6f);
  for (int d = threadIdx.x; d < width; d += threads) {
    const float normalized = round_activation((READ_A(start + d) - average) * inverse, dtype);
    const float scale = round_activation(1.f + READ_C(mod + d), dtype);
    WRITE_OUTPUT(start + d, round_activation(__fmul_rn(normalized, scale), dtype) + READ_B(mod + d));
  }
}
}  // namespace

void launch_flux_rotary_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t*,
                             const KernelParams& p) {
  if (p.output_count()) {
    flux_rotary_pairs<<<(unsigned(p.output_count() / 2) + 255) / 256, 256, 0, current_cuda_stream()>>>(out, a, b, p);
  }
}
void launch_flux_modulate_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                               const KernelParams& p) {
  if (p.output_count()) {
    flux_modulate_rows<<<p.output_count() / p.output_shape(2), threads, 0, current_cuda_stream()>>>(out, a, b, c, p);
  }
}
}  // namespace elman
