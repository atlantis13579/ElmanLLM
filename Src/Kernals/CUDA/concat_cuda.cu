#include "kernel.cuh"

namespace elman {
template<class Value>
__global__ void concat_float_rows(Value* output, const Value* a, const Value* b, int count, int left, int right) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(count)) return;
  const int width = left + right, row = int(i) / width, column = int(i) % width;
  output[i] = column < left ? a[row * left + column] : b[row * right + column - left];
}
__global__ void kernel_concat_cuda(uint32_t* out, const uint32_t* a,
                                   const uint32_t* b, const uint32_t* c,
                                   KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(operator_attr::concat::kAxis);
    int inner = p.input_stride(KernelInput::A, axis);
    int full = p.output_shape(axis);
    int coordinate = (i / inner) % full;
    int outer = i / (inner * full);
    int rest = i % inner;
    if (coordinate < p.input_shape(KernelInput::A, axis)) {
      int source = outer * p.input_shape(KernelInput::A, axis) * inner +
                   coordinate * inner + rest;
      COPY_INPUT_A(i, source);
    } else {
      int source = outer * p.input_shape(KernelInput::B, axis) * inner +
                   (coordinate - p.input_shape(KernelInput::A, axis)) * inner +
                   rest;
      if (p.input_dtype(KernelInput::B) == p.output_dtype()) {
        COPY_TYPED_ELEMENT(out, b, i, source, p.output_dtype());
      } else if (!kernel_integer(p.input_dtype(KernelInput::B))) {
        WRITE_OUTPUT(i, READ_B(source));
      } else {
        WRITE_INT_OUTPUT(i, READ_INT_B(source));
      }
    }
  }
}
void launch_concat_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                        const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    if (p.output_dtype() == int(DType::Float32) && p.input_dtype(KernelInput::A) == p.output_dtype() &&
        p.input_dtype(KernelInput::B) == p.output_dtype()) {
      const int axis = p.attribute(operator_attr::concat::kAxis), inner = p.input_stride(KernelInput::A, axis);
      const int left = p.input_shape(KernelInput::A, axis) * inner, right = p.input_shape(KernelInput::B, axis) * inner;
      if (left % 4 == 0 && right % 4 == 0 && reinterpret_cast<uintptr_t>(out) % 16 == 0 &&
          reinterpret_cast<uintptr_t>(a) % 16 == 0 && reinterpret_cast<uintptr_t>(b) % 16 == 0) {
        concat_float_rows<<<(unsigned(p.output_count() / 4) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
            reinterpret_cast<uint4*>(out), reinterpret_cast<const uint4*>(a), reinterpret_cast<const uint4*>(b),
            p.output_count() / 4, left / 4, right / 4);
      } else concat_float_rows<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
          out, a, b, p.output_count(), left, right);
      return;
    }
    kernel_concat_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
