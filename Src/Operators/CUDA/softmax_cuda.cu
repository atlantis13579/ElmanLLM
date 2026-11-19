#include "softmax_rows.cuh"

namespace elman {
struct SoftmaxInput {
  __device__ float operator()(const uint32_t* a, const uint32_t*, const KernelParams& p, int i) const {
    return READ_A(i);
  }
};
void launch_softmax_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         const uint32_t* c, const KernelParams& p) {
  launch_softmax_cuda_impl(out, a, nullptr, p, SoftmaxInput{}, p.input_dtype(KernelInput::A));
}
} // namespace elman
