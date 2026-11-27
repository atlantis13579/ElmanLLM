#include "kernel.h"

namespace elman {
void launch_avg_pool2d_grad_cpu(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int input_width = p.output_shape(tensor_axis::image::kWidth);
    int input_height = p.output_shape(tensor_axis::image::kHeight);
    int plane = i / (input_width * input_height);
    int iy = (i / input_width) % input_height;
    int ix = i % input_width;
    int output_height =
        p.input_shape(KernelInput::B, tensor_axis::image::kHeight);
    int output_width =
        p.input_shape(KernelInput::B, tensor_axis::image::kWidth);
    float value = 0.0f;
    for (int oy = 0; oy < output_height; ++oy) {
      for (int ox = 0; ox < output_width; ++ox) {
        int start_y = oy * p.attribute(kernal_attr::pool2d::kStrideHeight) -
                      p.attribute(kernal_attr::pool2d::kPaddingHeight);
        int start_x = ox * p.attribute(kernal_attr::pool2d::kStrideWidth) -
                      p.attribute(kernal_attr::pool2d::kPaddingWidth);
        if (iy >= start_y &&
            iy < start_y + p.attribute(kernal_attr::pool2d::kKernelHeight) &&
            ix >= start_x &&
            ix < start_x + p.attribute(kernal_attr::pool2d::kKernelWidth)) {
          int count = p.attribute(kernal_attr::pool2d::kKernelHeight) *
                      p.attribute(kernal_attr::pool2d::kKernelWidth);
          if (p.attribute(kernal_attr::pool2d::kCountIncludePadding) == 0) {
            count = 0;
            for (int ky = 0;
                 ky < p.attribute(kernal_attr::pool2d::kKernelHeight); ++ky) {
              for (int kx = 0;
                   kx < p.attribute(kernal_attr::pool2d::kKernelWidth);
                   ++kx) {
                int y = start_y + ky;
                int x = start_x + kx;
                if (y >= 0 && y < input_height && x >= 0 && x < input_width) {
                  ++count;
                }
              }
            }
          }
          if (count > 0) {
            value += READ_B((plane * output_height + oy) * output_width + ox) /
                     float(count);
          }
        }
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
