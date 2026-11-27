#include "kernel.cuh"

namespace elman {
__global__ void slice_grad_float_contiguous(uint32_t* output, const uint32_t* input,
                                           int count, int inner, int length, int source_length, int start, int step) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(count)) return;
  const int row = int(i) / inner;
  const int coordinate = row % length - start;
  const bool valid = coordinate >= 0 && coordinate % step == 0 && coordinate / step < source_length;
  output[i] = valid ? input[((row / length) * source_length + coordinate / step) * inner + int(i) % inner] : 0u;
}
__global__ void kernel_slice_grad_cuda(uint32_t* out, const uint32_t* a,
                                       const uint32_t* b, const uint32_t* c,
                                       KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int flat = i;
    int source = 0;
    int valid = 1;
    for (int axis = p.output_rank() - 1; axis >= 0; --axis) {
      int coordinate = flat % p.output_shape(axis);
      flat /= p.output_shape(axis);
      if (axis == p.attribute(kernal_attr::slice::kAxis)) {
        coordinate -= p.attribute(kernal_attr::slice::kStart);
        if (coordinate < 0 ||
            coordinate % p.attribute(kernal_attr::slice::kStep) != 0) {
          valid = 0;
        }
        coordinate /= p.attribute(kernal_attr::slice::kStep);
        if (coordinate >= p.input_shape(KernelInput::A, axis)) {
          valid = 0;
        }
      }
      source += coordinate * p.input_stride(KernelInput::A, axis);
    }
    WRITE_OUTPUT(i, valid != 0 ? READ_A(source) : 0.0f);
  }
}
void launch_slice_grad_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    if (p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == p.output_dtype()) {
      const int axis = p.attribute(kernal_attr::slice::kAxis);
      slice_grad_float_contiguous<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
          out, a, p.output_count(), p.input_stride(KernelInput::A, axis), p.output_shape(axis),
          p.input_shape(KernelInput::A, axis), p.attribute(kernal_attr::slice::kStart), p.attribute(kernal_attr::slice::kStep));
      return;
    }
    kernel_slice_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
