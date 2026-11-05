#include "Operators/conv2d_params.h"
#include "kernel.h"

namespace elman {
void launch_conv2d_input_grad_cpu(uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c,
                                  const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int input_width = p.output_shape(conv2d::kWidthAxis);
    int input_height = p.output_shape(conv2d::kHeightAxis);
    int input_channels = p.output_shape(conv2d::kChannelAxis);
    int ix = i % input_width;
    int iy = (i / input_width) % input_height;
    int ic = (i / (input_width * input_height)) % input_channels;
    int batch = i / (input_channels * input_height * input_width);
    int output_channels = p.input_shape(KernelInput::A, conv2d::kChannelAxis);
    int output_height = p.input_shape(KernelInput::A, conv2d::kHeightAxis);
    int output_width = p.input_shape(KernelInput::A, conv2d::kWidthAxis);
    int kernel_height = p.input_shape(KernelInput::B, conv2d::kHeightAxis);
    int kernel_width = p.input_shape(KernelInput::B, conv2d::kWidthAxis);
    int channels_per_group =
        p.input_shape(KernelInput::B, conv2d::kChannelAxis);
    int outputs_per_group = output_channels / p.attribute(conv2d::kGroups);
    int group = ic / channels_per_group;
    int local_channel = ic % channels_per_group;
    float value = 0.0f;
    for (int local_output = 0; local_output < outputs_per_group;
         ++local_output) {
      for (int ky = 0; ky < kernel_height; ++ky) {
        for (int kx = 0; kx < kernel_width; ++kx) {
          int numerator_y = iy + p.attribute(conv2d::kPaddingHeight) -
                            ky * p.attribute(conv2d::kDilationHeight);
          int numerator_x = ix + p.attribute(conv2d::kPaddingWidth) -
                            kx * p.attribute(conv2d::kDilationWidth);
          if (numerator_y >= 0 && numerator_x >= 0 &&
              numerator_y % p.attribute(conv2d::kStrideHeight) == 0 &&
              numerator_x % p.attribute(conv2d::kStrideWidth) == 0) {
            int oy = numerator_y / p.attribute(conv2d::kStrideHeight);
            int ox = numerator_x / p.attribute(conv2d::kStrideWidth);
            int oc = group * outputs_per_group + local_output;
            if (oy < output_height && ox < output_width) {
              value +=
                  READ_A(((batch * output_channels + oc) * output_height + oy) *
                             output_width +
                         ox) *
                  READ_B(((oc * channels_per_group + local_channel) *
                              kernel_height +
                          ky) *
                             kernel_width +
                         kx);
            }
          }
        }
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
