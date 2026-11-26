#include "Backends/CpuParallel.h"
#include "Core/LocalFunction.h"
#include "Kernals/conv2d_params.h"
#include "kernel.h"

namespace elman {
void launch_conv2d_input_grad_cpu(uint32_t* out, const uint32_t* a,
                                  const uint32_t* b, const uint32_t* c,
                                  const KernelParams& p) {
  parallel_for_cpu(p.output_count(), 128, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      int input_width = p.output_shape(operator_attr::conv2d::kWidthAxis);
      int input_height = p.output_shape(operator_attr::conv2d::kHeightAxis);
      int input_channels = p.output_shape(operator_attr::conv2d::kChannelAxis);
      int ix = i % input_width;
      int iy = (i / input_width) % input_height;
      int ic = (i / (input_width * input_height)) % input_channels;
      int batch = i / (input_channels * input_height * input_width);
      int output_channels = p.input_shape(KernelInput::A, operator_attr::conv2d::kChannelAxis);
      int output_height = p.input_shape(KernelInput::A, operator_attr::conv2d::kHeightAxis);
      int output_width = p.input_shape(KernelInput::A, operator_attr::conv2d::kWidthAxis);
      int kernel_height = p.input_shape(KernelInput::B, operator_attr::conv2d::kHeightAxis);
      int kernel_width = p.input_shape(KernelInput::B, operator_attr::conv2d::kWidthAxis);
      int channels_per_group =
          p.input_shape(KernelInput::B, operator_attr::conv2d::kChannelAxis);
      int outputs_per_group = output_channels / p.attribute(operator_attr::conv2d::kGroups);
      int group = ic / channels_per_group;
      int local_channel = ic % channels_per_group;
      const DType input_dtype = static_cast<DType>(p.input_dtype(KernelInput::A));
      if (is_narrow_float(input_dtype)) {
        // Store GEMM columns and each col2im update in
        // the input dtype. Preserve those rounding boundaries during overlap.
        LocalFunction round{[input_dtype](float v) {
          return input_dtype == DType::Float16 ? numeric::half_to_float(numeric::float_to_half(v)) :
                                                numeric::bfloat_to_float(numeric::float_to_bfloat(v));
        }};
        float value = 0;
        for (int ky = 0; ky < kernel_height; ++ky) {
          for (int kx = 0; kx < kernel_width; ++kx) {
            int ny = iy + p.attribute(operator_attr::conv2d::kPaddingHeight) - ky * p.attribute(operator_attr::conv2d::kDilationHeight);
            int nx = ix + p.attribute(operator_attr::conv2d::kPaddingWidth) - kx * p.attribute(operator_attr::conv2d::kDilationWidth);
            if (ny < 0 || nx < 0 || ny % p.attribute(operator_attr::conv2d::kStrideHeight) || nx % p.attribute(operator_attr::conv2d::kStrideWidth)) continue;
            int oy = ny / p.attribute(operator_attr::conv2d::kStrideHeight), ox = nx / p.attribute(operator_attr::conv2d::kStrideWidth);
            if (oy >= output_height || ox >= output_width) continue;
            float column = 0;
            for (int local_output = 0; local_output < outputs_per_group; ++local_output) {
              int oc = group * outputs_per_group + local_output;
              column += READ_A(((batch * output_channels + oc) * output_height + oy) * output_width + ox) *
                        READ_B(((oc * channels_per_group + local_channel) * kernel_height + ky) * kernel_width + kx);
            }
            value = round(value + round(column));
          }
        }
        WRITE_OUTPUT(i, value);
        continue;
      }
      float value = 0.0f;
      for (int local_output = 0; local_output < outputs_per_group;
           ++local_output) {
        for (int ky = 0; ky < kernel_height; ++ky) {
          for (int kx = 0; kx < kernel_width; ++kx) {
            int numerator_y = iy + p.attribute(operator_attr::conv2d::kPaddingHeight) -
                              ky * p.attribute(operator_attr::conv2d::kDilationHeight);
            int numerator_x = ix + p.attribute(operator_attr::conv2d::kPaddingWidth) -
                              kx * p.attribute(operator_attr::conv2d::kDilationWidth);
            if (numerator_y >= 0 && numerator_x >= 0 &&
                numerator_y % p.attribute(operator_attr::conv2d::kStrideHeight) == 0 &&
                numerator_x % p.attribute(operator_attr::conv2d::kStrideWidth) == 0) {
              int oy = numerator_y / p.attribute(operator_attr::conv2d::kStrideHeight);
              int ox = numerator_x / p.attribute(operator_attr::conv2d::kStrideWidth);
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
  });
}
}  // namespace elman
