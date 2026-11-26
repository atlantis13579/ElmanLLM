#include "kernel.h"
#include "Core/LocalFunction.h"
#include "Kernals/low_precision_softmax.h"
#include "Kernals/low_precision_pointwise_math.h"
#include <vector>
#include "low_precision_softmax_exp.h"

namespace elman {
namespace {
float softmax_row_sum(const std::vector<float>& values, bool contiguous) {
    if (!contiguous) {
        float total = 0;
        for (float value : values) total += value;
        return total;
    }
    // Match the FP32 AVX2 reduction lanes, including a partial final vector.
    float lanes[8]{};
    for (size_t i = 0; i < values.size(); ++i) {
        if (i < 8) lanes[i] = values[i];
        else lanes[i % 8] += values[i];
    }
    float total = lanes[0];
    for (size_t i = 1; i < std::min(size_t(8), values.size()); ++i) total += lanes[i];
    return total;
}
}
void launch_low_precision_softmax_cpu(LowPrecisionSoftmax op, uint32_t* output, const uint32_t* input,
                                     const uint32_t* upstream, const KernelParams& p) {
    const int count = p.output_count();
    if (!count) return;
    const int axis = p.attribute(0), mode = p.attribute(1);
    const int inner = p.input_rank(KernelInput::A) ? p.input_stride(KernelInput::A, axis) : 1;
    const int length = p.input_rank(KernelInput::A) ? p.input_shape(KernelInput::A, axis) : 1;
    const bool last = p.input_rank(KernelInput::A) == 0 || axis == p.input_rank(KernelInput::A) - 1;
    const bool logarithmic = op == LowPrecisionSoftmax::LogSoftmax;
    std::vector<float> values(length), terms(length), result(count);
    for (int row = 0; row < count / length; ++row) {
        const int base = row / inner * length * inner + row % inner;
        const bool vector = last || row % inner < inner / 8 * 8;
        LocalFunction exponential{[vector](float x) { return vector ? low_softmax_vector_exp(x) : std::exp(x); }};
        for (int j = 0; j < length; ++j)
            values[j] = low_precision_value(input, base + j * inner, DType(p.input_dtype(KernelInput::A)));
        if (mode == 2) {
            for (int j = 0; j < length; ++j) {
                const float g = low_precision_value(upstream, base + j * inner, DType(p.output_dtype()));
                terms[j] = logarithmic ? g : g * values[j];
            }
            const float sum = softmax_row_sum(terms, last);
            for (int j = 0; j < length; ++j) {
                const float g = low_precision_value(upstream, base + j * inner, DType(p.output_dtype()));
                result[base + j * inner] = logarithmic ? g - exponential(values[j]) * sum : (g - sum) * values[j];
            }
        } else {
            float maximum = -std::numeric_limits<float>::infinity();
            for (float value : values) maximum = std::isnan(value) || value > maximum ? value : maximum;
            for (int j = 0; j < length; ++j) terms[j] = exponential(values[j] - maximum);
            const float sum = softmax_row_sum(terms, last);
            const float normalizer = logarithmic ? (vector ? low_softmax_vector_log(sum) : std::log(sum)) : 1.f / sum;
            for (int j = 0; j < length; ++j)
                result[base + j * inner] = logarithmic ? values[j] - maximum - normalizer :
                    last ? terms[j] * normalizer : terms[j] / sum;
        }
    }
    if (p.output_dtype() == int(DType::Float32)) { std::memcpy(output, result.data(), size_t(count) * sizeof(float)); return; }
    uint8_t* bytes = reinterpret_cast<uint8_t*>(output);
    const DType dtype = DType(p.output_dtype());
    if (dtype == DType::Float4E2M1FN) {
        for (int i = 0; i < count; i += 2)
            bytes[i / 2] = uint8_t(low_precision_encode(result[i], dtype) |
                (i + 1 < count ? low_precision_encode(result[i + 1], dtype) << 4 : 0));
    } else for (int i = 0; i < count; ++i) bytes[i] = low_precision_encode(result[i], dtype);
}
} // namespace elman
