#include "kernel.h"

namespace elman {
void launch_batch_norm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                           const uint32_t* c, const KernelParams& p) {
  for (int i = 0; i < p.output_count(); ++i) {
    int channels = p.input_shape(KernelInput::A, tensor_axis::image::kChannels);
    int spatial = p.input_stride(KernelInput::A, tensor_axis::image::kChannels);
    int channel = (i / spatial) % channels;
    WRITE_OUTPUT(i, (READ_A(i) - READ_B(channel)) /
                        EL_SQRT(READ_C(channel) +
                                float_attribute(
                                    p, kernal_attr::batch_norm::kEpsilon)));
  }
}
}  // namespace elman
