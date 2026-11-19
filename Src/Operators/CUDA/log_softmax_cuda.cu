#include "kernel.cuh"
#include "row_reduce.cuh"

namespace elman {
namespace {
constexpr int threads = 128;

template<bool Maximum>
__device__ float warp_total(float value) {
  for (int offset = 16; offset; offset >>= 1) {
    const float other = __shfl_xor_sync(0xffffffff, value, offset);
    value = Maximum ? fmaxf(value, other) : value + other;
  }
  return value;
}

template<bool Maximum>
__device__ float block_total(float value, float* partial) {
  const int lane = threadIdx.x % 32;
  value = warp_total<Maximum>(value);
  if (!lane) partial[threadIdx.x / 32] = value;
  __syncthreads();
  return warp_total<Maximum>(lane < blockDim.x / 32 ? partial[lane] : (Maximum ? -EL_INF : 0.f));
}

// Keep each row in registers across the max, exponential sum and write.
template<int Vectors>
__global__ void log_softmax_float4(float4* out, const float4* a, int row_vectors) {
  __shared__ float maxima[32], sums[32];
  const int base = int(blockIdx.x) * row_vectors;
  float4 values[Vectors];
  float maximum = -EL_INF;
  #pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    const int col = threadIdx.x + j * blockDim.x;
    auto v = col < row_vectors ? a[base + col] : make_float4(-EL_INF,-EL_INF,-EL_INF,-EL_INF);
    values[j] = v;
    maximum = fmaxf(maximum, fmaxf(fmaxf(v.x,v.y), fmaxf(v.z,v.w)));
  }
  maximum = block_total<true>(maximum, maxima);
  float sum = 0;
  #pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    auto& v = values[j];
    v.x -= maximum; v.y -= maximum; v.z -= maximum; v.w -= maximum;
    sum += expf(v.x) + expf(v.y) + expf(v.z) + expf(v.w);
  }
  const float normalizer = logf(block_total<false>(sum, sums));
  #pragma unroll
  for (int j = 0; j < Vectors; ++j) {
    const int col = threadIdx.x + j * blockDim.x;
    const auto v = values[j];
    if (col < row_vectors)
      out[base + col] = make_float4(v.x-normalizer,v.y-normalizer,v.z-normalizer,v.w-normalizer);
  }
}

__global__ void log_softmax_rows(uint32_t* out, const uint32_t* a, int length, KernelParams p) {
  __shared__ float scratch[threads];
  const int base = int(blockIdx.x) * length;
  float maximum = -EL_INF;
  for (int column = threadIdx.x; column < length; column += threads)
    maximum = fmaxf(maximum, READ_A(base + column));
  maximum = row_reduce<true>(maximum, scratch);
  float sum = 0;
  for (int column = threadIdx.x; column < length; column += threads)
    sum += EL_EXP(READ_A(base + column) - maximum);
  const float log_sum = EL_LOG(row_reduce<false>(sum, scratch));
  for (int column = threadIdx.x; column < length; column += threads) {
    const int i = base + column;
    WRITE_OUTPUT(i, READ_A(i) - maximum - log_sum);
  }
}

__global__ void log_softmax_strided(uint32_t* out, const uint32_t* a,
                                    int inner, int length, int rows, KernelParams p) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= unsigned(rows)) return;
  const int base = (row / inner) * inner * length + row % inner;
  float maximum = -EL_INF;
  for (int column = 0; column < length; ++column)
    maximum = EL_MAX(maximum, READ_A(base + column * inner));
  float sum = 0;
  for (int column = 0; column < length; ++column)
    sum += EL_EXP(READ_A(base + column * inner) - maximum);
  const float log_sum = EL_LOG(sum);
  for (int column = 0; column < length; ++column) {
    const int i = base + column * inner;
    WRITE_OUTPUT(i, READ_A(i) - maximum - log_sum);
  }
}
}  // namespace

void launch_log_softmax_cuda(uint32_t* out, const uint32_t* a,
                             const uint32_t*, const uint32_t*, const KernelParams& p) {
  if (!p.output_count()) return;
  const int axis = p.attribute(operator_attr::softmax::kAxis);
  const int rank = p.input_rank(KernelInput::A);
  const int inner = rank ? p.input_stride(KernelInput::A, axis) : 1;
  const int length = rank ? p.input_shape(KernelInput::A, axis) : 1;
  const int rows = p.output_count() / length;
  const auto stream = current_cuda_stream();
  if (inner == 1 && length >= 128 && length <= 4096 && length % 4 == 0 &&
      p.input_dtype(KernelInput::A) == int(DType::Float32) && p.output_dtype() == int(DType::Float32) &&
      (reinterpret_cast<uintptr_t>(a) | reinterpret_cast<uintptr_t>(out)) % alignof(float4) == 0) {
    auto dst = reinterpret_cast<float4*>(out);
    auto src = reinterpret_cast<const float4*>(a);
    if (length <= 1024)
      log_softmax_float4<1><<<rows, ((length / 4 + 31) / 32) * 32, 0, stream>>>(dst, src, length / 4);
    else
      log_softmax_float4<4><<<rows, 256, 0, stream>>>(dst, src, length / 4);
    return;
  }
  if (inner == 1 && length >= 64)
    log_softmax_rows<<<rows, threads, 0, stream>>>(out, a, length, p);
  else
    log_softmax_strided<<<(unsigned(rows) + threads - 1) / threads, threads, 0, stream>>>(
        out, a, inner, length, rows, p);
}
}  // namespace elman
