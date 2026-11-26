#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Kernals/group_norm_params.h"
#include <limits>

namespace elman {
void launch_group_norm_weight_grad_cpu(uint32_t* out, const uint32_t* a,
                                       const uint32_t* b, const uint32_t*, const KernelParams& p) {
    int batch = p.input_shape(KernelInput::A, 0), channels = p.input_shape(KernelInput::A, 1);
    int spatial = p.input_stride(KernelInput::A, 1), width = channels / p.attribute(0);
    int length = width * spatial;
    LocalFunction rounded{[&](float value) {
        if (p.output_dtype() == int(DType::Float16)) return numeric::half_to_float(numeric::float_to_half(value));
        if (p.output_dtype() == int(DType::BFloat16)) return numeric::bfloat_to_float(numeric::float_to_bfloat(value));
        return value;
    }};
    for (int channel = 0; channel < channels; ++channel) {
        if (!spatial) { WRITE_OUTPUT(channel, batch ? std::numeric_limits<float>::quiet_NaN() : 0.0f); continue; }
        float result = 0;
        for (int row = 0; row < batch; ++row) {
            int base = (row * channels + channel / width * width) * spatial;
            float mean = 0, variance = 0;
            // Welford preserves the small-row rounding used by the reference
            // before statistics are narrowed to the affine parameter dtype.
            for (int j = 0; j < length; ++j) {
                float value = READ_A(base + j), delta = value - mean;
                mean += delta / float(j + 1);
                variance += delta * (value - mean);
            }
            float inverse = rounded(float(1.0 / std::sqrt(double(variance / float(length)) + group_norm_epsilon(p))));
            mean = rounded(mean);
            float product = 0, gradient = 0;
            int start = (row * channels + channel) * spatial;
            for (int j = 0; j < spatial; ++j) {
                float dy = READ_B(start + j);
                product += dy * READ_A(start + j);
                gradient += dy;
            }
            result += (product - gradient * mean) * inverse;
        }
        WRITE_OUTPUT(channel, result);
    }
}
} // namespace elman
