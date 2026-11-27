#include "kernel.cuh"
#include "long_row_reduction.cuh"
#include "integer_reduction.cuh"

namespace elman {
__global__ void kernel_prod_cuda(uint32_t* out, const uint32_t* a,
                                 const uint32_t* b, const uint32_t* c,
                                 KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(kernal_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = axis < 0 ? input_count(KernelInput::A, p)
                          : p.input_shape(KernelInput::A, axis);
    int base = axis < 0 ? 0 : (i / inner) * length * inner + i % inner;
    if (kernel_integer(p.output_dtype())) {
      uint32_t integer_value = 1u;
      for (int j = 0; j < length; ++j) {
        integer_value *= uint32_t(READ_INT_A(base + j * inner));
      }
      WRITE_INT_OUTPUT(i, int32_t(integer_value));
      return;
    }
    float value = 1.0f;
    for (int j = 0; j < length; ++j) {
      float x = READ_A(base + j * inner);
      value = value * x;
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_prod_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                      const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    if (p.output_dtype() == int(DType::Int64)) {
      integer_reduction_cuda<true><<<p.output_count(), 256, 0, current_cuda_stream()>>>(reinterpret_cast<uint64_t*>(out), a, p);
      return;
    }
    if (try_reduce_contiguous_rows<RowReduction::Product>(out, a, p)) return;
    kernel_prod_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
