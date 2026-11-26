#include "kernel.h"
#include "Kernals/low_precision_categorical.h"
#include "Kernals/low_precision_pointwise_math.h"

namespace elman {
void launch_low_precision_categorical_cpu(uint32_t* output, const uint32_t* weights, const uint32_t* cumulative, const uint32_t* uniforms, const KernelParams& p) {
    const int width = p.input_shape(KernelInput::A, 1);
    const DType weight_dtype = DType(p.input_dtype(KernelInput::A));
    const DType cumulative_dtype = DType(p.input_dtype(KernelInput::B));
    const DType uniform_dtype = DType(p.input_dtype(KernelInput::C));
    for (int row = 0; row < p.output_count(); ++row) {
        const float uniform = low_precision_value(uniforms, row, uniform_dtype);
        const float total = low_precision_value(cumulative, row * width + width - 1, cumulative_dtype);
        bool invalid = !std::isfinite(uniform) || uniform < 0.f || uniform >= 1.f || !std::isfinite(total) || total <= 0.f;
        const float threshold = std::min(uniform * total, std::nextafter(total, 0.f));
        int chosen = -1, last = -1;
        for (int j = 0; j < width; ++j) {
            const float weight = low_precision_value(weights, row * width + j, weight_dtype);
            const float value = low_precision_value(cumulative, row * width + j, cumulative_dtype);
            invalid |= !std::isfinite(weight) || weight < 0.f || !std::isfinite(value) || value < 0.f;
            if (weight > 0.f) {
                last = j;
                if (chosen < 0 && value > threshold) chosen = j;
            }
        }
        output[row] = uint32_t(invalid ? -1 : chosen < 0 ? last : chosen);
    }
}
} // namespace elman
