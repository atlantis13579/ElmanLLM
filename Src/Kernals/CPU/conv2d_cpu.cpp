#include "Backends/CpuParallel.h"
#include "Kernals/conv2d_params.h"
#include "kernel.h"

namespace elman {
void launch_conv2d_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                       const uint32_t* c, const KernelParams& p) {
  parallel_for_cpu(p.output_count(), 128, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      int output_width = p.output_shape(operator_attr::conv2d::kWidthAxis);
      int output_height = p.output_shape(operator_attr::conv2d::kHeightAxis);
      int output_channels = p.output_shape(operator_attr::conv2d::kChannelAxis);
      int ox = i % output_width;
      int oy = (i / output_width) % output_height;
      int oc = (i / (output_width * output_height)) % output_channels;
      int batch = i / (output_channels * output_height * output_width);
      int input_channels = p.input_shape(KernelInput::A, operator_attr::conv2d::kChannelAxis);
      int input_height = p.input_shape(KernelInput::A, operator_attr::conv2d::kHeightAxis);
      int input_width = p.input_shape(KernelInput::A, operator_attr::conv2d::kWidthAxis);
      int kernel_height = p.input_shape(KernelInput::B, operator_attr::conv2d::kHeightAxis);
      int kernel_width = p.input_shape(KernelInput::B, operator_attr::conv2d::kWidthAxis);
      int channels_per_group =
          p.input_shape(KernelInput::B, operator_attr::conv2d::kChannelAxis);
      int outputs_per_group = output_channels / p.attribute(operator_attr::conv2d::kGroups);
      int first_channel = (oc / outputs_per_group) * channels_per_group;
      float value = p.input_rank(KernelInput::C) >= 0 ? READ_C(oc) : 0.0f;
      for (int ic = 0; ic < channels_per_group; ++ic) {
        for (int ky = 0; ky < kernel_height; ++ky) {
          for (int kx = 0; kx < kernel_width; ++kx) {
            int iy = oy * p.attribute(operator_attr::conv2d::kStrideHeight) -
                     p.attribute(operator_attr::conv2d::kPaddingHeight) +
                     ky * p.attribute(operator_attr::conv2d::kDilationHeight);
            int ix = ox * p.attribute(operator_attr::conv2d::kStrideWidth) -
                     p.attribute(operator_attr::conv2d::kPaddingWidth) +
                     kx * p.attribute(operator_attr::conv2d::kDilationWidth);
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
  });
}
}  // namespace elman
