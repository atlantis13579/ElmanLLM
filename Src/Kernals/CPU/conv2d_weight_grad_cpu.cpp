#include "Backends/CpuParallel.h"
#include "Kernals/conv2d_params.h"
#include "kernel.h"

namespace elman {
void launch_conv2d_weight_grad_cpu(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   const KernelParams& p) {
  parallel_for_cpu(p.output_count(), 16, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      int kernel_width = p.output_shape(kernal_attr::conv2d::kWidthAxis);
      int kernel_height = p.output_shape(kernal_attr::conv2d::kHeightAxis);
      int channels_per_group = p.output_shape(kernal_attr::conv2d::kChannelAxis);
      int output_channels = p.output_shape(kernal_attr::conv2d::kOutputChannelAxis);
      int kx = i % kernel_width;
      int ky = (i / kernel_width) % kernel_height;
      int local_channel =
          (i / (kernel_width * kernel_height)) % channels_per_group;
      int oc = i / (channels_per_group * kernel_height * kernel_width);
      int batches = p.input_shape(KernelInput::A, kernal_attr::conv2d::kBatchAxis);
      int input_channels = p.input_shape(KernelInput::A, kernal_attr::conv2d::kChannelAxis);
      int input_height = p.input_shape(KernelInput::A, kernal_attr::conv2d::kHeightAxis);
      int input_width = p.input_shape(KernelInput::A, kernal_attr::conv2d::kWidthAxis);
      int output_height = p.input_shape(KernelInput::B, kernal_attr::conv2d::kHeightAxis);
      int output_width = p.input_shape(KernelInput::B, kernal_attr::conv2d::kWidthAxis);
      int outputs_per_group = output_channels / p.attribute(kernal_attr::conv2d::kGroups);
      int ic = (oc / outputs_per_group) * channels_per_group + local_channel;
      float value = 0.0f;
      for (int batch = 0; batch < batches; ++batch) {
        float batch_value = 0.0f;
        for (int oy = 0; oy < output_height; ++oy) {
          for (int ox = 0; ox < output_width; ++ox) {
            int iy = oy * p.attribute(kernal_attr::conv2d::kStrideHeight) -
                     p.attribute(kernal_attr::conv2d::kPaddingHeight) +
                     ky * p.attribute(kernal_attr::conv2d::kDilationHeight);
            int ix = ox * p.attribute(kernal_attr::conv2d::kStrideWidth) -
                     p.attribute(kernal_attr::conv2d::kPaddingWidth) +
                     kx * p.attribute(kernal_attr::conv2d::kDilationWidth);
            if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
              batch_value +=
                  READ_A(((batch * input_channels + ic) * input_height + iy) *
                             input_width +
                         ix) *
                  READ_B(((batch * output_channels + oc) * output_height + oy) *
                             output_width +
                         ox);
            }
          }
        }
        value += batch_value;
        const DType dtype = static_cast<DType>(p.input_dtype(KernelInput::A));
        // The CPU fallback accumulates batch GEMMs into a narrow output buffer.
        if (dtype == DType::Float16) value = numeric::half_to_float(numeric::float_to_half(value));
        if (dtype == DType::BFloat16) value = numeric::bfloat_to_float(numeric::float_to_bfloat(value));
      }
      WRITE_OUTPUT(i, value);
    }
  });
}
}  // namespace elman
