#include "kernel.h"
#include <cstring>
#include "Operators/low_precision_reduction_math.h"

namespace elman {
namespace {
float low_cpu_reduction_value(LowPrecisionReduction op, int i, const uint32_t* a, const uint32_t* b,
                              const uint32_t* c, const KernelParams& p) {
    float value = low_precision_reduction_result(op, i, a, b, c, p);
    if ((op != LowPrecisionReduction::Amax && op != LowPrecisionReduction::Amin) || value != 0.f) return value;
    const int length = low_precision_reduction_length(p), inner = low_precision_reduction_inner(p);
    const int base = p.attribute(0) < 0 ? 0 : i / inner * length * inner + i % inner;
    // AVX2 reduction groups four eight-lane vectors. Its vector
    // comparison retains the last equal zero; scalar tails retain the first.
    if (inner > 1 && i % inner < inner / 32 * 32) {
        for (int j = 0; j < length; ++j) {
            const float x = low_precision_value(a, base + j * inner, DType(p.input_dtype(KernelInput::A)));
            if (x == 0.f) value = x;
        }
    } else if (inner == 1 && length >= 32) {
        int64_t selected = -1;
        for (int j = 0; j < length / 32 * 32; ++j) {
            const float x = low_precision_value(a, base + j, DType(p.input_dtype(KernelInput::A)));
            const int64_t priority = (int64_t(7 - j % 8) * 4 + (j / 8) % 4) * (int64_t(length) + 1) + j;
            if (x == 0.f && priority > selected) { selected = priority; value = x; }
        }
    }
    return value;
}
void store_low_reduction(uint32_t* output, int index, float value, DType dtype) {
    if (dtype == DType::Float32) { reinterpret_cast<float*>(output)[index] = value; return; }
    auto* bytes = reinterpret_cast<uint8_t*>(output);
    const uint8_t code = low_precision_encode(value, dtype);
    if (dtype != DType::Float4E2M1FN) bytes[index] = code;
    else {
        const int shift = (index & 1) * 4;
        bytes[index / 2] = uint8_t((bytes[index / 2] & ~(15 << shift)) | (code << shift));
    }
}
}
void launch_low_precision_reduction_cpu(LowPrecisionReduction op, uint32_t* out, const uint32_t* a,
                                        const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    const DType dtype = DType(p.output_dtype()), input_dtype = DType(p.input_dtype(KernelInput::A));
    const int count = p.output_count();
    if (!count) return;
    if (dtype == DType::Float4E2M1FN) std::memset(out, 0, (size_t(count) + 1) / 2);
    if (op == LowPrecisionReduction::Cumsum || op == LowPrecisionReduction::ReverseCumsum) {
        const int length = low_precision_reduction_length(p), inner = low_precision_reduction_inner(p);
        const bool reverse = op == LowPrecisionReduction::ReverseCumsum;
        for (int row = 0; row < count / length; ++row) {
            const int base = row / inner * length * inner + row % inner;
            double value = 0.;
            for (int j = 0; j < length; ++j) {
                const int i = base + (reverse ? length - 1 - j : j) * inner;
                const float x = low_precision_value(a, i, input_dtype);
                value += double(x);
                store_low_reduction(out, i, p.input_rank(KernelInput::A) == 0 || (reverse && length == 1) ? x : float(value), dtype);
            }
        }
    } else {
        for (int i = 0; i < count; ++i)
            store_low_reduction(out, i, low_cpu_reduction_value(op, i, a, b, c, p), dtype);
    }
}
} // namespace elman
