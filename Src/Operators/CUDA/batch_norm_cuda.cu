#include "kernel.cuh"

namespace elman {
__global__ void kernel_batch_norm_cuda(uint32_t* out, const uint32_t* a,
                                       const uint32_t* b, const uint32_t* c,
                                       KernelParams p) {
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;
  if (element < unsigned(p.output_count())) {
    int i = int(element);
    int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
    int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
    int channel = (i / spatial) % channels;
    WRITE_OUTPUT(i, (READ_A(i) - READ_B(channel)) /
                        EL_SQRT(READ_C(channel) +
                                float_attribute(
                                    p, operator_attr::batch_norm::kEpsilon)));
  }
}
void launch_batch_norm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                            const uint32_t* c, const KernelParams& p) {
  if (p.output_count() > 0) {
    kernel_batch_norm_cuda<<<(unsigned(p.output_count()) + 255u) / 256u, 256>>>(
        out, a, b, c, p);
  }
}
}  // namespace elman
