#include "kernel.h"

namespace elman {
void launch_max_pool2d_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int output_width = p.output_shape(tensor_axis::image::kWidth);
    int output_height = p.output_shape(tensor_axis::image::kHeight);
    int plane = i / (output_width * output_height);
    int oy = (i / output_width) % output_height;
    int ox = i % output_width;
    int input_height =
        p.input_shape(KernelInput::A, tensor_axis::image::kHeight);
    int input_width = p.input_shape(KernelInput::A, tensor_axis::image::kWidth);
    float value = -EL_INF;
    int count = 0;
    for (int ky = 0; ky < p.attribute(operator_attr::pool2d::kKernelHeight);
         ++ky) {
      for (int kx = 0; kx < p.attribute(operator_attr::pool2d::kKernelWidth);
           ++kx) {
        int iy = oy * p.attribute(operator_attr::pool2d::kStrideHeight) -
                 p.attribute(operator_attr::pool2d::kPaddingHeight) + ky;
        int ix = ox * p.attribute(operator_attr::pool2d::kStrideWidth) -
                 p.attribute(operator_attr::pool2d::kPaddingWidth) + kx;
        if (iy >= 0 && iy < input_height && ix >= 0 && ix < input_width) {
          float x = READ_A((plane * input_height + iy) * input_width + ix);
          value = EL_MAX(value, x);
          ++count;
        } else if (p.attribute(operator_attr::pool2d::kCountIncludePadding) !=
                   0) {
          ++count;
        }
      }
    }
    WRITE_OUTPUT(i, value);
  }
}
}  // namespace elman
