#include "Operators/conv2d_params.h"
#include "kernel.cuh"

namespace elman {
__global__ void kernel_conv2d_cuda(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int output_width = p.output_shape(conv2d::kWidthAxis);
    int output_height = p.output_shape(conv2d::kHeightAxis);
    int output_channels = p.output_shape(conv2d::kChannelAxis);
    int ox = i % output_width;
    int oy = (i / output_width) % output_height;
    int oc = (i / (output_width * output_height)) % output_channels;
    int batch = i / (output_channels * output_height * output_width);
    int input_channels = p.input_shape(KernelInput::A, conv2d::kChannelAxis);
    int input_height = p.input_shape(KernelInput::A, conv2d::kHeightAxis);
    int input_width = p.input_shape(KernelInput::A, conv2d::kWidthAxis);
    int kernel_height = p.input_shape(KernelInput::B, conv2d::kHeightAxis);
    int kernel_width = p.input_shape(KernelInput::B, conv2d::kWidthAxis);
    int channels_per_group =
        p.input_shape(KernelInput::B, conv2d::kChannelAxis);
    int outputs_per_group = output_channels / p.attribute(conv2d::kGroups);
    int first_channel = (oc / outputs_per_group) * channels_per_group;
    float value = p.input_rank(KernelInput::C) >= 0 ? READ_C(oc) : 0.0f;
    for (int ic = 0; ic < channels_per_group; ++ic) {
      for (int ky = 0; ky < kernel_height; ++ky) {
        for (int kx = 0; kx < kernel_width; ++kx) {
          int iy = oy * p.attribute(conv2d::kStrideHeight) -
                   p.attribute(conv2d::kPaddingHeight) +
                   ky * p.attribute(conv2d::kDilationHeight);
          int ix = ox * p.attribute(conv2d::kStrideWidth) -
                   p.attribute(conv2d::kPaddingWidth) +
                   kx * p.attribute(conv2d::kDilationWidth);
          if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
            value +=
                READ_A(((batch * input_channels + first_channel + ic) *
                            input_height +
                        iy) *
                           input_width +
                       ix) *
                READ_B(((oc * channels_per_group + ic) * kernel_height + ky) *
                           kernel_width +
                       kx);
          }
        }
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_conv2d_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_conv2d_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
