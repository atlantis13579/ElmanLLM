#include "Operators/float64_categorical.h"
#include <algorithm>
#include <cmath>

namespace elman {
void launch_float64_categorical_cpu(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& p) {
    const auto* a = reinterpret_cast<const double*>(weights);
    const auto* b = reinterpret_cast<const double*>(cumulative);
    const auto* c = reinterpret_cast<const double*>(uniforms);
    const int width = p.input_shape(KernelInput::A, 1);
    for (int row = 0; row < p.output_count(); ++row) {
        const double uniform = c[row], total = b[row * width + width - 1];
        bool invalid = !std::isfinite(uniform) || uniform < 0. || uniform >= 1. || !std::isfinite(total) || total <= 0.;
        const double threshold = std::min(uniform * total, std::nextafter(total, 0.));
        int chosen = -1, last = -1;
        for (int j = 0; j < width; ++j) {
            const double weight = a[row * width + j], value = b[row * width + j];
            invalid |= !std::isfinite(weight) || weight < 0. || !std::isfinite(value) || value < 0.;
            if (weight > 0.) {
                last = j;
                if (chosen < 0 && value > threshold) chosen = j;
            }
        }
        output[row] = uint32_t(invalid ? -1 : chosen < 0 ? last : chosen);
    }
}
} // namespace elman
