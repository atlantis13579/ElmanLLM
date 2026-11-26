#include "Backends/CudaConvolution.h"
#include "Backends/CudaGemm.h"
#include "Core/AllocationPool.h"
#include "Core/Memory.h"
#include "Kernals/conv2d_params.h"
#include "kernel.cuh"

namespace elman {
namespace {
__global__ void kernel_conv2d_weight_grad_cuda(uint32_t* out, const uint32_t* a,
                                               const uint32_t* b,
                                               const uint32_t* c,
                                               KernelParams p, bool repair_nan = false) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    if (repair_nan && !isnan(reinterpret_cast<float*>(out)[i])) return;
    int kernel_width = p.output_shape(operator_attr::conv2d::kWidthAxis);
    int kernel_height = p.output_shape(operator_attr::conv2d::kHeightAxis);
    int channels_per_group = p.output_shape(operator_attr::conv2d::kChannelAxis);
    int output_channels = p.output_shape(operator_attr::conv2d::kOutputChannelAxis);
    int kx = i % kernel_width;
    int ky = (i / kernel_width) % kernel_height;
    int local_channel =
        (i / (kernel_width * kernel_height)) % channels_per_group;
    int oc = i / (channels_per_group * kernel_height * kernel_width);
    int batches = p.input_shape(KernelInput::A, operator_attr::conv2d::kBatchAxis);
    int input_channels = p.input_shape(KernelInput::A, operator_attr::conv2d::kChannelAxis);
    int input_height = p.input_shape(KernelInput::A, operator_attr::conv2d::kHeightAxis);
    int input_width = p.input_shape(KernelInput::A, operator_attr::conv2d::kWidthAxis);
    int output_height = p.input_shape(KernelInput::B, operator_attr::conv2d::kHeightAxis);
    int output_width = p.input_shape(KernelInput::B, operator_attr::conv2d::kWidthAxis);
    int outputs_per_group = output_channels / p.attribute(operator_attr::conv2d::kGroups);
    int ic = (oc / outputs_per_group) * channels_per_group + local_channel;
    float value = 0.0f;
    for (int batch = 0; batch < batches; ++batch) {
      for (int oy = 0; oy < output_height; ++oy) {
        for (int ox = 0; ox < output_width; ++ox) {
          int iy = oy * p.attribute(operator_attr::conv2d::kStrideHeight) -
                   p.attribute(operator_attr::conv2d::kPaddingHeight) +
                   ky * p.attribute(operator_attr::conv2d::kDilationHeight);
          int ix = ox * p.attribute(operator_attr::conv2d::kStrideWidth) -
                   p.attribute(operator_attr::conv2d::kPaddingWidth) +
                   kx * p.attribute(operator_attr::conv2d::kDilationWidth);
          if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
            value +=
                READ_A(((batch * input_channels + ic) * input_height + iy) *
                           input_width +
                       ix) *
                READ_B(((batch * output_channels + oc) * output_height + oy) *
                           output_width +
                       ox);
          }
        }
      }
    }
    WRITE_OUTPUT(i, value);
  }
}

constexpr int filter_tile = 128;
// Pack input patches and upstream values in reduction tiles. Each GEMM owns
// its partial result so the final FP32 reduction has a fixed order.
__global__ void pack_filter_tiles(float* packed, const float* input, const float* grad, int count,
                                  int inputs, int outputs, int height, int width, int positions) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  const int patches = inputs * 9, rows = patches + outputs;
  const int row = i / filter_tile % rows, pos = i / (filter_tile * rows) * filter_tile + i % filter_tile;
  float value = 0.f;
  if (pos < positions) {
    const int spatial = height * width, batch = pos / spatial, point = pos % spatial;
    if (row >= patches) value = grad[(batch * outputs + row - patches) * spatial + point];
    else {
      const int x = point % width + row % 3 - 1, y = point / width + row / 3 % 3 - 1;
      if (unsigned(x) < unsigned(width) && unsigned(y) < unsigned(height))
        value = input[((batch * inputs + row / 9) * height + y) * width + x];
    }
  }
  packed[i] = value;
}
__global__ void sum_filter_tiles(float* out, const float* partials, int weights, int tiles) {
  const int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= weights) return;
  float sum = partials[i];
  for (int tile = 1; tile < tiles; ++tile) sum += partials[tile * weights + i];
  out[i] = sum;
}
bool tiled_filter_gradient(uint32_t* out, const uint32_t* a, const uint32_t* b, const KernelParams& p) {
  using namespace operator_attr::conv2d;
  constexpr int fp32 = int(DType::Float32);
  if (p.output_rank() != 4 || p.input_rank(KernelInput::A) != 4 || p.input_rank(KernelInput::B) != 4 ||
      p.output_dtype() != fp32 || p.input_dtype(KernelInput::A) != fp32 || p.input_dtype(KernelInput::B) != fp32 ||
      p.attribute(kGroups) != 1 || p.output_shape(kHeightAxis) != 3 || p.output_shape(kWidthAxis) != 3 ||
      p.attribute(kStrideHeight) != 1 || p.attribute(kStrideWidth) != 1 || p.attribute(kPaddingHeight) != 1 ||
      p.attribute(kPaddingWidth) != 1 || p.attribute(kDilationHeight) != 1 || p.attribute(kDilationWidth) != 1)
    return false;
  const int inputs = p.output_shape(kChannelAxis), outputs = p.output_shape(kOutputChannelAxis);
  const int height = p.input_shape(KernelInput::A,kHeightAxis), width = p.input_shape(KernelInput::A,kWidthAxis);
  const int batch = p.input_shape(KernelInput::A,kBatchAxis);
  if (inputs != p.input_shape(KernelInput::A,kChannelAxis) || outputs != p.input_shape(KernelInput::B,kChannelAxis) ||
      batch != p.input_shape(KernelInput::B,kBatchAxis) || height != p.input_shape(KernelInput::B,kHeightAxis) ||
      width != p.input_shape(KernelInput::B,kWidthAxis)) return false;
  const int positions = batch * height * width, weights = p.output_count();
  if (positions < 1024 || weights < 4096) return false;
  const int tiles = int((unsigned(positions) + filter_tile - 1) / filter_tile);
  const size_t packed_count = size_t(tiles) * filter_tile * (size_t(inputs) * 9 + outputs);
  const size_t elements = packed_count + size_t(tiles) * weights;
  if (elements > (32u << 20) / sizeof(float)) return false;
  BufferPtr workspace;
  try { workspace = backend_workspace(Backend::CUDA, elements * sizeof(float)); }
  catch (const MemoryLimitExceeded&) { return false; }
  catch (const AllocationFailure&) { return false; }
  float* packed = static_cast<float*>(workspace->raw_pointer());
  float* partials = packed + packed_count;
  cudaStream_t stream = current_cuda_stream();
  pack_filter_tiles<<<(unsigned(packed_count) + 255) / 256, 256, 0, stream>>>(packed,
      reinterpret_cast<const float*>(a), reinterpret_cast<const float*>(b), int(packed_count), inputs, outputs, height, width, positions);
  convolution_filter_partials_cuda(partials, packed, inputs * 9, outputs, filter_tile, tiles);
  const unsigned blocks = (unsigned(weights) + 255) / 256;
  sum_filter_tiles<<<blocks, 256, 0, stream>>>(reinterpret_cast<float*>(out), partials, weights, tiles);
  // Skip padded taps even when an upstream infinity made zero times infinity NaN.
  kernel_conv2d_weight_grad_cuda<<<blocks, 256, 0, stream>>>(out, a, b, nullptr, p, true);
  return true;
}
} // namespace
void launch_conv2d_weight_grad_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    const KernelParams& p) {
  if (p.output_count() > 0 &&
      !try_pointwise_conv_gradient_cuda(CudaConvolution::WeightGradient, out, a, b, p) &&
      !tiled_filter_gradient(out, a, b, p) &&
      !try_convolution_cuda(CudaConvolution::WeightGradient, out, a, b, p)) {
    kernel_conv2d_weight_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                                     256, 0, current_cuda_stream()>>>(out, a, b, c, p);
  }
}
}  // namespace elman
