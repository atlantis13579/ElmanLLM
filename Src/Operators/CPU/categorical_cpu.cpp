#include "kernel.h"

namespace elman {
void launch_categorical_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    int width = p.input_shape(KernelInput::A, 1);
    for (int row = 0; row < p.output_count(); ++row) {
        float uniform = READ_C(row), total = READ_B(row * width + width - 1);
        bool invalid = !std::isfinite(uniform) || uniform < 0 || uniform >= 1 || !std::isfinite(total) || total <= 0;
        float threshold = std::min(uniform * total, std::nextafter(total, 0.0f));
        int chosen = -1, last = -1;
        for (int j = 0; j < width; ++j) {
            float weight = READ_A(row * width + j), cumulative = READ_B(row * width + j);
            invalid |= !std::isfinite(weight) || weight < 0 || !std::isfinite(cumulative) || cumulative < 0;
            if (weight > 0) {
                last = j;
                if (chosen < 0 && cumulative > threshold) chosen = j;
            }
        }
        out[row] = uint32_t(invalid ? -1 : chosen < 0 ? last : chosen);
    }
}
} // namespace elman
