#include "kernel.cuh"

namespace elman {
__global__ void kernel_log_softmax_cuda(uint32_t* out, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c,
                                        KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(operator_attr::softmax::kAxis);
    int inner = p.input_stride(KernelInput::A, axis);
    int length = p.input_shape(KernelInput::A, axis);
    int base = (i / (length * inner)) * length * inner + i % inner;
    float maximum_value = -EL_INF;
    for (int j = 0; j < length; ++j) {
      maximum_value = EL_MAX(maximum_value, READ_A(base + j * inner));
    }
    float denominator = 0.0f;
    for (int j = 0; j < length; ++j) {
      denominator += EL_EXP(READ_A(base + j * inner) - maximum_value);
    }
    WRITE_OUTPUT(i, READ_A(i) - maximum_value - EL_LOG(denominator));
  }
}
void launch_log_softmax_cuda(uint32_t* out, const uint32_t* a,
                             const uint32_t* b, const uint32_t* c,
                             const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_log_softmax_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                              256>>>(out, a, b, c, p);
  }
}
}  // namespace elman
