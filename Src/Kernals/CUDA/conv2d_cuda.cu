#include "Backends/CudaGemm.h"
#include "Backends/CudaConvolution.h"
#include "Core/AllocationPool.h"
#include "Kernals/conv2d_params.h"
#include "kernel.cuh"

#include <algorithm>

namespace elman {
namespace {
using namespace operator_attr::conv2d;

__device__ float direct_value(const uint32_t* a, const uint32_t* b, const uint32_t* c,
                              const KernelParams& p, int i) {
  const int output_width = p.output_shape(kWidthAxis), output_height = p.output_shape(kHeightAxis);
  const int output_channels = p.output_shape(kChannelAxis);
  const int spatial = output_width * output_height;
  const int ox = i % output_width, oy = (i / output_width) % output_height;
  const int oc = (i / spatial) % output_channels, batch = i / (output_channels * spatial);
  const int input_channels = p.input_shape(KernelInput::A, kChannelAxis);
  const int input_height = p.input_shape(KernelInput::A, kHeightAxis);
  const int input_width = p.input_shape(KernelInput::A, kWidthAxis);
  const int kernel_height = p.input_shape(KernelInput::B, kHeightAxis);
  const int kernel_width = p.input_shape(KernelInput::B, kWidthAxis);
  const int channels_per_group = p.input_shape(KernelInput::B, kChannelAxis);
  const int outputs_per_group = output_channels / p.attribute(kGroups);
  const int first_channel = (oc / outputs_per_group) * channels_per_group;
  const int top = oy * p.attribute(kStrideHeight) - p.attribute(kPaddingHeight);
  const int left = ox * p.attribute(kStrideWidth) - p.attribute(kPaddingWidth);
  float value = p.input_rank(KernelInput::C) >= 0 ? READ_C(oc) : 0.0f;
  for (int ic = 0; ic < channels_per_group; ++ic) {
    for (int ky = 0; ky < kernel_height; ++ky) {
      const int iy = top + ky * p.attribute(kDilationHeight);
      for (int kx = 0; kx < kernel_width; ++kx) {
        const int ix = left + kx * p.attribute(kDilationWidth);
        if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
          value += READ_A(((batch * input_channels + first_channel + ic) * input_height + iy) * input_width + ix) *
                   READ_B(((oc * channels_per_group + ic) * kernel_height + ky) * kernel_width + kx);
        }
      }
    }
  }
  return value;
}

__global__ void direct_conv2d(uint32_t* out, const uint32_t* a, const uint32_t* b,
                              const uint32_t* c, KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < unsigned(p.output_count())) WRITE_OUTPUT(i, direct_value(a, b, c, p, i));
}

template <typename Word>
__global__ void unfold_conv2d(Word* columns, const Word* input, int start, int positions, KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  const int channels = p.input_shape(KernelInput::A, kChannelAxis);
  const int kernel_width = p.input_shape(KernelInput::B, kWidthAxis);
  const int kernel_height = p.input_shape(KernelInput::B, kHeightAxis);
  if (i >= unsigned(positions) * channels * kernel_height * kernel_width) return;
  const int row = i / positions, position = start + int(i % positions);
  const int output_width = p.output_shape(kWidthAxis);
  const int spatial = output_width * p.output_shape(kHeightAxis);
  const int input_width = p.input_shape(KernelInput::A, kWidthAxis);
  const int input_height = p.input_shape(KernelInput::A, kHeightAxis);
  const int y = position % spatial / output_width * p.attribute(kStrideHeight) - p.attribute(kPaddingHeight)
              + row / kernel_width % kernel_height * p.attribute(kDilationHeight);
  const int x = position % output_width * p.attribute(kStrideWidth) - p.attribute(kPaddingWidth)
              + row % kernel_width * p.attribute(kDilationWidth);
  const int plane = position / spatial * channels + row / (kernel_width * kernel_height);
  columns[i] = y >= 0 && y < input_height && x >= 0 && x < input_width
      ? input[(plane * input_height + y) * input_width + x] : Word(0);
}

__global__ void finish_conv2d(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                              const float* product, int batch_offset, int spatial, int start,
                              int positions, bool padded, KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(positions) * p.output_shape(kChannelAxis)) return;
  const int channel = i / positions;
  const int position = start + int(i % positions);
  const int index = batch_offset + (position / spatial * p.output_shape(kChannelAxis) + channel) * spatial + position % spatial;
  const float value = product ? product[i] + READ_C(channel)
                              : elman_cuda_decode(elman_cuda_load(out, index, p.output_dtype()), p.output_dtype());
  // A zero in the patch matrix can multiply a nonfinite weight. The direct
  // convolution skips those padded taps, so recompute such results.
  if (padded && isnan(value)) WRITE_OUTPUT(index, direct_value(a, b, c, p, index));
  else if (product) WRITE_OUTPUT(index, value);
}

bool try_unfold_conv2d(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  const DType dtype = DType(p.output_dtype());
  const int groups = p.attribute(kGroups);
  if (!is_compute_float(dtype) || groups <= 0 ||
      p.input_dtype(KernelInput::A) != p.output_dtype() || p.input_dtype(KernelInput::B) != p.output_dtype())
    return false;

  const int batches = p.output_shape(kBatchAxis);
  const int channels = p.output_shape(kChannelAxis);
  const int spatial = p.output_shape(kHeightAxis) * p.output_shape(kWidthAxis);
  const int input_channels = p.input_shape(KernelInput::A, kChannelAxis);
  if (groups > 1 && (input_channels % groups || channels % groups || input_channels / groups < 4 ||
                     channels / groups < 4 || p.input_shape(KernelInput::B, kChannelAxis) != input_channels / groups))
    return false;
  if (!input_channels || spatial < 64 || (channels < 16 && spatial < 4096)) return false;
  const int input_spatial = p.input_shape(KernelInput::A, kHeightAxis) * p.input_shape(KernelInput::A, kWidthAxis);
  const int kernel_area = p.input_shape(KernelInput::B, kHeightAxis) * p.input_shape(KernelInput::B, kWidthAxis);
  const int shared = input_channels * kernel_area;
  if (!input_spatial || shared < 16) return false;

  const bool padded = p.attribute(kPaddingHeight) || p.attribute(kPaddingWidth);
  const bool unfold = kernel_area != 1 || padded || p.attribute(kStrideHeight) != 1 || p.attribute(kStrideWidth) != 1;
  const bool biased = p.input_rank(KernelInput::C) >= 0;
  const size_t word_bytes = dtype_storage_bytes(dtype, 1);
  const size_t column_bytes = unfold ? size_t(shared) * word_bytes : 0;
  const size_t bytes_per_position = column_bytes + (biased ? size_t(channels) * sizeof(float) : 0);
  constexpr size_t workspace_limit = 32 * 1024 * 1024;
  constexpr int tile_alignment = 128;
  // A dense column tile can span NCHW batch boundaries when the epilogue
  // already scatters the product to its final channel and spatial position.
  const int batch_step = unfold && biased ? batches : 1;
  const int positions = spatial * batch_step;
  int tile = positions;
  if (bytes_per_position && size_t(positions) > workspace_limit / bytes_per_position) {
    tile = int(workspace_limit / bytes_per_position / tile_alignment) * tile_alignment;
    if (!tile) return false;
  }
  BufferPtr workspace;
  uint8_t* columns = nullptr;
  float* product = nullptr;
  if (bytes_per_position) {
    const size_t columns_size = tile * column_bytes;
    const size_t product_offset = (columns_size + alignof(float) - 1) & ~(alignof(float) - 1);
    const size_t workspace_size = biased ? product_offset + size_t(tile) * channels * sizeof(float) : columns_size;
    workspace = backend_workspace(Backend::CUDA, workspace_size);
    uint8_t* storage = static_cast<uint8_t*>(workspace->raw_pointer());
    columns = unfold ? storage : nullptr;
    product = biased ? reinterpret_cast<float*>(storage + product_offset) : nullptr;
  }

  const cudaStream_t stream = current_cuda_stream();
  const uint8_t* input = reinterpret_cast<const uint8_t*>(a);
  uint8_t* output = reinterpret_cast<uint8_t*>(out);
  for (int batch = 0; batch < batches; batch += batch_step) {
    const uint8_t* input_batch = input + size_t(batch) * input_channels * input_spatial * word_bytes;
    const int batch_offset = batch * channels * spatial;
    for (int start = 0; start < positions;) {
      const int count = std::min(tile, positions - start);
      if (unfold) {
        const unsigned blocks = (unsigned(count) * shared + 255u) / 256u;
        if (word_bytes == sizeof(uint16_t))
          unfold_conv2d<<<blocks, 256, 0, stream>>>(reinterpret_cast<uint16_t*>(columns),
              reinterpret_cast<const uint16_t*>(input_batch), start, count, p);
        else
          unfold_conv2d<<<blocks, 256, 0, stream>>>(reinterpret_cast<uint32_t*>(columns),
              reinterpret_cast<const uint32_t*>(input_batch), start, count, p);
      }
      const void* patches = unfold ? columns : input_batch + size_t(start) * word_bytes;
      void* destination = biased ? static_cast<void*>(product) : output + size_t(batch_offset + start) * word_bytes;
      if (!try_conv2d_gemm_cuda(destination, patches, b, dtype, count, channels / groups, shared / groups,
                               unfold ? count : input_spatial, biased ? count : spatial, biased, groups))
        return false;
      if (biased || padded)
        finish_conv2d<<<(unsigned(count) * channels + 255u) / 256u, 256, 0, stream>>>(
            out, a, b, c, product, batch_offset, spatial, start, count, padded, p);
      start += count;
    }
  }
  return true;
}

__global__ void finish_native_conv(float* out, const uint32_t* a, const uint32_t* b,
                                   const uint32_t* c, bool padded, KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(p.output_count())) return;
  const int spatial = p.output_shape(kHeightAxis) * p.output_shape(kWidthAxis);
  const int channel = i / spatial % p.output_shape(kChannelAxis);
  const float value = p.input_rank(KernelInput::C) >= 0 ? out[i] + READ_C(channel) : out[i];
  out[i] = padded && isnan(value) ? direct_value(a, b, c, p, i) : value;
}
}  // namespace

void launch_conv2d_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (!p.output_count()) return;
  // Small dense filters map directly to a compact GEMM. Grouped filters
  // and wide receptive fields benefit from a specialized convolution plan.
  const int kernel_area = p.input_shape(KernelInput::B, kHeightAxis) * p.input_shape(KernelInput::B, kWidthAxis);
  const bool grouped = p.attribute(kGroups) > 1 && p.input_shape(KernelInput::B, kChannelAxis) > 1;
  if ((grouped || kernel_area >= 25) && try_convolution_cuda(CudaConvolution::Forward, out, a, b, p)) {
    const bool padded = p.attribute(kPaddingHeight) || p.attribute(kPaddingWidth);
    if (padded || p.input_rank(KernelInput::C) >= 0)
      finish_native_conv<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
          reinterpret_cast<float*>(out), a, b, c, padded, p);
    return;
  }
  if (!try_unfold_conv2d(out, a, b, c, p))
    direct_conv2d<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, a, b, c, p);
}
}  // namespace elman
