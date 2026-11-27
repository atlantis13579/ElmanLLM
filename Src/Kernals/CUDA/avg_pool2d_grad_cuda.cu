#include "kernel.cuh"

namespace elman {
__global__ void kernel_avg_pool2d_grad_cuda(uint32_t* out, const uint32_t* a,
                                            const uint32_t* b,
                                            const uint32_t* c, KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
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
    const int stride_y = p.attribute(kernal_attr::pool2d::kStrideHeight);
    const int stride_x = p.attribute(kernal_attr::pool2d::kStrideWidth);
    const int kernel_y = p.attribute(kernal_attr::pool2d::kKernelHeight);
    const int kernel_x = p.attribute(kernal_attr::pool2d::kKernelWidth);
    const int y = iy + p.attribute(kernal_attr::pool2d::kPaddingHeight);
    const int x = ix + p.attribute(kernal_attr::pool2d::kPaddingWidth);
    const int first_y = y < kernel_y ? 0 : (y - kernel_y) / stride_y + 1;
    const int first_x = x < kernel_x ? 0 : (x - kernel_x) / stride_x + 1;
    const int stop_y = min(output_height, y / stride_y + 1);
    const int stop_x = min(output_width, x / stride_x + 1);
    for (int oy = first_y; oy < stop_y; ++oy) {
      for (int ox = first_x; ox < stop_x; ++ox) {
        int start_y = oy * p.attribute(kernal_attr::pool2d::kStrideHeight) -
                      p.attribute(kernal_attr::pool2d::kPaddingHeight);
        int start_x = ox * p.attribute(kernal_attr::pool2d::kStrideWidth) -
                      p.attribute(kernal_attr::pool2d::kPaddingWidth);
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
    WRITE_OUTPUT(i, value);
  }
}
void launch_avg_pool2d_grad_cuda(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c,
                                 const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_avg_pool2d_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                                  256, 0, current_cuda_stream()>>>(out, a, b, c, p);
  }
}
}  // namespace elman
