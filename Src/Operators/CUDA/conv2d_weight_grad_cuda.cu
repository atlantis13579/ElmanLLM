#include "Operators/conv2d_params.h"
#include "kernel.cuh"

namespace elman {
__global__ void kernel_conv2d_weight_grad_cuda(uint32_t* out, const uint32_t* a,
                                               const uint32_t* b,
                                               const uint32_t* c,
                                               KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int kernel_width = p.output_shape(conv2d::kWidthAxis);
    int kernel_height = p.output_shape(conv2d::kHeightAxis);
    int channels_per_group = p.output_shape(conv2d::kChannelAxis);
    int output_channels = p.output_shape(conv2d::kOutputChannelAxis);
    int kx = i % kernel_width;
    int ky = (i / kernel_width) % kernel_height;
    int local_channel =
        (i / (kernel_width * kernel_height)) % channels_per_group;
    int oc = i / (channels_per_group * kernel_height * kernel_width);
    int batches = p.input_shape(KernelInput::A, conv2d::kBatchAxis);
    int input_channels = p.input_shape(KernelInput::A, conv2d::kChannelAxis);
    int input_height = p.input_shape(KernelInput::A, conv2d::kHeightAxis);
    int input_width = p.input_shape(KernelInput::A, conv2d::kWidthAxis);
    int output_height = p.input_shape(KernelInput::B, conv2d::kHeightAxis);
    int output_width = p.input_shape(KernelInput::B, conv2d::kWidthAxis);
    int outputs_per_group = output_channels / p.attribute(conv2d::kGroups);
    int ic = (oc / outputs_per_group) * channels_per_group + local_channel;
    float value = 0.0f;
    for (int batch = 0; batch < batches; ++batch) {
      for (int oy = 0; oy < output_height; ++oy) {
        for (int ox = 0; ox < output_width; ++ox) {
          int iy = oy * p.attribute(conv2d::kStrideHeight) -
                   p.attribute(conv2d::kPaddingHeight) +
                   ky * p.attribute(conv2d::kDilationHeight);
          int ix = ox * p.attribute(conv2d::kStrideWidth) -
                   p.attribute(conv2d::kPaddingWidth) +
                   kx * p.attribute(conv2d::kDilationWidth);
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
void launch_conv2d_weight_grad_cuda(uint32_t* out, const uint32_t* a,
                                    const uint32_t* b, const uint32_t* c,
                                    const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_conv2d_weight_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                                     256>>>(out, a, b, c, p);
  }
}
}  // namespace elman
