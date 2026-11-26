#include "Core/AllocationPool.h"
#include "kernel.cuh"

namespace elman {
namespace {
constexpr int threads = 256;
constexpr int tile_elements = 4096;
constexpr int warp_threads = 32;

// A running mean and centered sum of squares can be merged across tiles.
struct Moments {
  float mean, squared;
  int count;
  __device__ void add(float value) {
    const float delta = value - mean;
    mean += delta / float(++count);
    squared += delta * (value - mean);
  }
};

__device__ Moments merge_moments(Moments left, Moments right) {
  if (!left.count) return right;
  if (!right.count) return left;
  const float delta = right.mean - left.mean;
  const int count = left.count + right.count;
  const float right_fraction = float(right.count) / float(count);
  return {left.mean + delta * right_fraction,
          left.squared + right.squared + delta * delta * (float(left.count) * right_fraction), count};
}

__device__ Moments warp_moments(Moments moments, int lane) {
  for (int offset = warp_threads / 2; offset > 0; offset /= 2) {
    const Moments other{__shfl_down_sync(0xffffffff, moments.mean, offset),
                        __shfl_down_sync(0xffffffff, moments.squared, offset),
                        __shfl_down_sync(0xffffffff, moments.count, offset)};
    if (lane < offset) moments = merge_moments(moments, other);
  }
  return moments;
}

__device__ Moments block_moments(Moments moments) {
  __shared__ Moments warps[threads / warp_threads];
  const int lane = threadIdx.x % warp_threads;
  moments = warp_moments(moments, lane);
  if (lane == 0) warps[threadIdx.x / warp_threads] = moments;
  __syncthreads();
  if (threadIdx.x < warp_threads)
    moments = warp_moments(lane < threads / warp_threads ? warps[lane] : Moments{}, lane);
  return moments;
}

__device__ float2 normalization_stats(float mean, float variance, const KernelParams& p) {
  const int dtype = p.input_dtype(KernelInput::A);
  const bool narrow = is_narrow_float(DType(dtype));
  float epsilon = float_attribute(p, operator_attr::group_norm::kEpsilon);
  if (narrow) epsilon = elman_cuda_decode(elman_cuda_encode(epsilon, dtype), dtype);
  float inverse = rsqrtf(variance + epsilon);
  if (narrow) {
    mean = elman_cuda_decode(elman_cuda_encode(mean, dtype), dtype);
    inverse = elman_cuda_decode(elman_cuda_encode(inverse, dtype), dtype);
  }
  return make_float2(mean, inverse);
}

__global__ void collect_moments(const uint32_t* a, Moments* partial, float2* stats, int length,
                                int tiles, KernelParams p) {
  const int group = blockIdx.x / tiles, start = int(blockIdx.x % tiles) * tile_elements;
  const int base = group * length + start;
  const int count = min(tile_elements, length - start);
  Moments moments{};
  for (int i = threadIdx.x; i < count; i += threads) moments.add(READ_A(base + i));
  moments = block_moments(moments);
  if (threadIdx.x == 0) {
    if (tiles == 1) stats[group] = normalization_stats(moments.mean, moments.squared / float(length), p);
    else partial[blockIdx.x] = moments;
  }
}

__global__ void merge_group_moments(const Moments* partial, float2* stats, int tiles, KernelParams p) {
  Moments moments{};
  for (int tile = threadIdx.x; tile < tiles; tile += threads)
    moments = merge_moments(moments, partial[size_t(blockIdx.x) * tiles + tile]);
  moments = block_moments(moments);
  if (threadIdx.x == 0)
    stats[blockIdx.x] = normalization_stats(moments.mean, moments.squared / float(moments.count), p);
}

__device__ float2 direct_group_stats(const uint32_t* a, int base, int length, const KernelParams& p) {
  float average = 0, variance = 0;
  for (int j = 0; j < length; ++j) average += READ_A(base + j);
  average /= float(length);
  for (int j = 0; j < length; ++j) {
    const float difference = READ_A(base + j) - average;
    variance += difference * difference;
  }
  return normalization_stats(average, variance / float(length), p);
}

__global__ void apply_group_norm(uint32_t* out, const uint32_t* a, const uint32_t* b,
                                 const uint32_t* c, const float2* stats, int spatial, int length,
                                 KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(p.output_count())) return;
  const int group = i / length;
  const float2 moments = stats ? stats[group] : direct_group_stats(a, group * length, length, p);
  const int channel = (i / spatial) % p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
  const float weight = p.input_rank(KernelInput::B) >= 0 ? READ_B(channel) : 1.0f;
  const float bias = p.input_rank(KernelInput::C) >= 0 ? READ_C(channel) : 0.0f;
  const float scale = moments.y * weight;
  const float value = spatial == 1 ? (READ_A(i) - moments.x) * moments.y * weight + bias
                                   : scale * READ_A(i) + (-scale * moments.x + bias);
  WRITE_OUTPUT(i, value);
}
}  // namespace

void launch_group_norm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  if (p.output_count() <= 0) return;
  const int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
  const int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
  const int length = (channels / p.attribute(operator_attr::group_norm::kGroups)) * spatial;
  const cudaStream_t stream = current_cuda_stream();
  BufferPtr workspace;
  float2* stats = nullptr;
  if (length > 32) {
    const int groups = p.output_count() / length;
    const int tiles = (length - 1) / tile_elements + 1;
    const size_t partial_bytes = tiles > 1 ? size_t(groups) * tiles * sizeof(Moments) : 0;
    workspace = backend_workspace(Backend::CUDA, size_t(groups) * sizeof(float2) + partial_bytes);
    stats = static_cast<float2*>(workspace->raw_pointer());
    Moments* partial = reinterpret_cast<Moments*>(stats + groups);
    collect_moments<<<groups * tiles, threads, 0, stream>>>(a, partial, stats, length, tiles, p);
    if (tiles > 1) merge_group_moments<<<groups, threads, 0, stream>>>(partial, stats, tiles, p);
  }
  apply_group_norm<<<(unsigned(p.output_count()) + threads - 1) / threads, threads, 0, stream>>>(
      out, a, b, c, stats, spatial, length, p);
}
}  // namespace elman
