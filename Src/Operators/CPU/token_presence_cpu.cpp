#include "kernel.h"

namespace elman {
void launch_token_presence_cpu(uint32_t* out, const uint32_t* a, const uint32_t*, const uint32_t*, const KernelParams& p) {
    if (!p.output_count()) return;
    std::fill(out, out + p.output_count(), 0u);
    int stride = p.output_shape(1), length = p.input_shape(KernelInput::A, 1);
    for (int row = 0; row < p.output_shape(0); ++row)
        for (int j = 0; j < length; ++j) {
            int id = READ_INT_A(row * length + j);
            out[row * stride + (id >= 0 && id < stride - 1 ? id : stride - 1)] = 1;
        }
}
} // namespace elman
