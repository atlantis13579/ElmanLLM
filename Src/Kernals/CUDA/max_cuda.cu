#include "kernel.cuh"
#include "long_row_reduction.cuh"
#include "integer_extrema.cuh"

namespace elman {
__global__ void kernel_max_cuda(uint32_t* out, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c,
                                KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(operator_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length = axis < 0 ? input_count(KernelInput::A, p)
                          : p.input_shape(KernelInput::A, axis);
    int base = axis < 0 ? 0 : (i / inner) * length * inner + i % inner;
    float value = -EL_INF;
    for (int j = 0; j < length; ++j) {
      float x = READ_A(base + j * inner);
      value = EL_MAX(value, x);
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_max_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                     const uint32_t* c, const KernelParams& p) {
  if (is_standard_integer(DType(p.output_dtype()))) {
    if (p.output_count() > 0)
      integer_extrema_cuda<true><<<unsigned(p.output_count()), 256, 0, current_cuda_stream()>>>(out, a, p);
    return;
  }
  if (p.output_count() > 0) {
    if (try_reduce_contiguous_rows<RowReduction::Maximum>(out, a, p)) return;
    kernel_max_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
