#include "Backends/CpuMath.h"
#include <immintrin.h>
#include <sleef.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace elman {
std::string vector_math_version_avx2() {
    return "SLEEF " + std::to_string(SLEEF_VERSION_MAJOR) + "." + std::to_string(SLEEF_VERSION_MINOR) + "." +
        std::to_string(SLEEF_VERSION_PATCHLEVEL) + " AVX2";
}
namespace {
__m256 evaluate(__m256 value, CpuMathOperation operation) {
    switch (operation) {
    case CpuMathOperation::Exp: return Sleef_expf8_u10avx2(value);
    case CpuMathOperation::Sin: return Sleef_sinf8_u10avx2(value);
    case CpuMathOperation::Cos: return Sleef_cosf8_u10avx2(value);
    case CpuMathOperation::Silu:
        return _mm256_div_ps(value, _mm256_add_ps(_mm256_set1_ps(1),
            Sleef_expf8_u10avx2(_mm256_xor_ps(value, _mm256_set1_ps(-0.0f)))));
    }
    return value;
}
float scalar(float value, CpuMathOperation operation) {
    switch (operation) {
    case CpuMathOperation::Exp: return std::exp(value);
    case CpuMathOperation::Sin: return std::sin(value);
    case CpuMathOperation::Cos: return std::cos(value);
    case CpuMathOperation::Silu: return value / (1.0f + std::exp(-value));
    }
    return value;
}
float sum(const float* input, int count) {
    float lanes[8]{};
    int initial = std::min(8, count);
    std::copy_n(input, initial, lanes);
    for (int i = 8; i < count; ++i) lanes[i % 8] += input[i];
    if (count < 8) {
        float total = lanes[0];
        for (int i = 1; i < count; ++i) total += lanes[i];
        return total;
    }
#if defined(__GNUC__) && __GNUC__ > 5 && !defined(_MSC_VER)
    for (int stride = 4; stride > 0; stride /= 2)
        for (int i = 0; i < stride; ++i) lanes[i] += lanes[i + stride];
    return lanes[0];
#else
    float total = lanes[0];
    for (int i = 1; i < 8; ++i) total += lanes[i];
    return total;
#endif
}
}

void unary_math_avx2(float* output, const float* input, int count, CpuMathOperation operation) {
    int i = 0;
    if (operation != CpuMathOperation::Silu) {
        for (; i + 8 <= count; i += 8)
            _mm256_storeu_ps(output + i, evaluate(_mm256_loadu_ps(input + i), operation));
        if (i < count) {
            float tail[8]{};
            std::copy_n(input + i, count - i, tail);
            _mm256_storeu_ps(tail, evaluate(_mm256_loadu_ps(tail), operation));
            std::copy_n(tail, count - i, output + i);
        }
        return;
    }
    // Elementwise TensorIterator processes two vectors, then a scalar tail.
    for (; i + 16 <= count; i += 16) {
        _mm256_storeu_ps(output + i, evaluate(_mm256_loadu_ps(input + i), operation));
        _mm256_storeu_ps(output + i + 8, evaluate(_mm256_loadu_ps(input + i + 8), operation));
    }
    for (; i < count; ++i) output[i] = scalar(input[i], operation);
}

void softmax_math_avx2(float* output, const float* input, int rows, int columns, bool narrow_input) {
    for (int row = 0; row < rows; ++row) {
        const float* source = input + row * columns;
        float* target = output + row * columns;
        float maximum = -std::numeric_limits<float>::infinity();
        for (int i = 0; i < columns; ++i)
            maximum = std::isnan(maximum) || maximum > source[i] ? maximum : source[i];
        int i = 0;
        for (; i + 8 <= columns; i += 8) {
            auto values = _mm256_sub_ps(_mm256_loadu_ps(source + i), _mm256_set1_ps(maximum));
            _mm256_storeu_ps(target + i, Sleef_expf8_u10avx2(values));
        }
        if (i < columns && !narrow_input) {
            float tail[8]{};
            std::copy_n(source + i, columns - i, tail);
            auto values = _mm256_sub_ps(_mm256_loadu_ps(tail), _mm256_set1_ps(maximum));
            _mm256_storeu_ps(tail, Sleef_expf8_u10avx2(values));
            std::copy_n(tail, columns - i, target + i);
        }
        float total = sum(target, narrow_input ? i : columns);
        // Narrow-input CPU softmax reduces complete FP32 vectors first and
        // computes the scalar tail with std::exp before the final reciprocal.
        if (narrow_input) for (; i < columns; ++i) {
            target[i] = std::exp(source[i] - maximum);
            total += target[i];
        }
        float inverse = 1.0f / total;
        for (int j = 0; j < columns; ++j) target[j] *= inverse;
    }
}
} // namespace elman
