#include "kernel.cuh"

namespace elman {
__global__ void kernel_reduce_grad_cuda(uint32_t* out, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c,
                                        KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int axis = p.attribute(kernal_attr::reduction::kAxis);
    int inner = axis < 0 ? 1 : p.input_stride(KernelInput::A, axis);
    int length =
        axis < 0 ? p.output_count() : p.input_shape(KernelInput::A, axis);
    int target = axis < 0 ? 0 : (i / (length * inner)) * inner + i % inner;
    float value = READ_C(target);
    int kind = p.attribute(kernal_attr::reduction::kGradientKind);
    if (kind == 1) {
      value = __fmul_rn(value, __fdiv_rn(1.0f, float(length)));
    }
    if (kind == 2 || kind == 3) {
      float extremum = READ_B(target);
      int base =
          axis < 0 ? 0 : (target / inner) * length * inner + target % inner;
      int matches = 0;
      for (int j = 0; j < length; ++j) {
        if (READ_A(base + j * inner) == extremum) {
          ++matches;
        }
      }
      value = (value / float(matches)) * float(READ_A(i) == extremum);
    }
    if (kind == 5 || kind == 6) {
      const float extremum = READ_B(target);
      const int base =
          axis < 0 ? 0 : (target / inner) * length * inner + target % inner;
      int selected = -1;
      for (int j = 0; j < length; ++j) {
        const int index = base + j * inner;
        const float candidate = READ_A(index);
        if (candidate == extremum || (candidate != candidate && extremum != extremum)) {
          selected = index;
          break;
        }
      }
      value = i == selected ? value : 0.0f;
    }
    if (kind == 4) {
      int base =
          axis < 0 ? 0 : (target / inner) * length * inner + target % inner;
      float product = 1.0f;
      for (int j = 0; j < length; ++j) {
        if (base + j * inner != i) {
          product *= READ_A(base + j * inner);
        }
      }
      value *= READ_A(i) == 0.0f ? product : READ_B(target) / READ_A(i);
    }
    WRITE_OUTPUT(i, value);
  }
}
void launch_reduce_grad_cuda(uint32_t* out, const uint32_t* a,
                             const uint32_t* b, const uint32_t* c,
                             const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_reduce_grad_cuda<<<(unsigned(p.output_count()) + 255u) / 256u,
                              256, 0, current_cuda_stream()>>>(out, a, b, c, p);
  }
}
}  // namespace elman
