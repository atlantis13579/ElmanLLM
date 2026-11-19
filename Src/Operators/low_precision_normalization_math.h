#pragma once
#include "Operators/low_precision_normalization.h"
#include "Operators/low_precision_pointwise_math.h"
#ifdef __CUDACC__
#define ELMAN_LOW_NORM_HD __device__
#else
#define ELMAN_LOW_NORM_HD
#endif
namespace elman {
ELMAN_LOW_NORM_HD inline float low_norm_mad(float a, float b, float c) {
#ifdef __CUDACC__
    return __fmaf_rn(a, b, c);
#else
    return a * b + c;
#endif
}
ELMAN_LOW_NORM_HD inline float low_norm_mul(float a, float b) {
#ifdef __CUDACC__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
struct LowNormLayout {
    LowPrecisionNormalization operation;
    int rows, width, channels, spatial, groups, mode;
    bool unbroadcasted, rounded, training, has_weight, has_bias;
    float epsilon;
    double wide_epsilon;
    DType dtype;
    ELMAN_LOW_NORM_HD LowNormLayout(LowPrecisionNormalization op, const KernelParams& p) : operation(low_precision_normalization_base(op)), mode(low_precision_normalization_mode(op, p)), dtype(DType(p.output_dtype())) {
        op = operation;
        training = op == LowPrecisionNormalization::Batch && p.attribute(9) == 1;
        has_weight = p.input_rank(KernelInput::B) >= 0;
        has_bias = p.input_rank(KernelInput::C) >= 0;
        const int rank = p.input_rank(KernelInput::A);
        unbroadcasted = rank == p.attribute(0);
        rounded = op == LowPrecisionNormalization::Rms && p.attribute(2) == 1;
        const bool trailing = op == LowPrecisionNormalization::Layer || op == LowPrecisionNormalization::Rms;
        wide_epsilon = low_precision_normalization_epsilon(op, p); epsilon = float(wide_epsilon);
        channels = rank >= 2 ? p.input_shape(KernelInput::A, 1) : 1;
        spatial = rank >= 2 ? p.input_stride(KernelInput::A, 1) : 1;
        groups = op == LowPrecisionNormalization::Group ? p.attribute(0) : 1;
        rows = 1; width = 1;
        if (trailing) {
            for (int d = 0; d < rank; ++d)
                if (d < rank - p.attribute(0)) rows *= p.input_shape(KernelInput::A, d);
                else width *= p.input_shape(KernelInput::A, d);
        } else if (op == LowPrecisionNormalization::Batch) { rows = channels; width = p.input_shape(KernelInput::A, 0) * spatial; }
        else { rows = p.input_shape(KernelInput::A, 0) * groups; width = channels / groups * spatial; }
    }
    ELMAN_LOW_NORM_HD int index(int row, int j) const {
        return operation == LowPrecisionNormalization::Batch ? (j / spatial * channels + row) * spatial + j % spatial : row * width + j;
    }
    ELMAN_LOW_NORM_HD int weight(int row, int j) const {
        if (operation == LowPrecisionNormalization::Batch) return row;
        if (operation == LowPrecisionNormalization::Group) return row % groups * (channels / groups) + j / spatial;
        return j;
    }
};
ELMAN_LOW_NORM_HD inline float low_norm_value(const uint32_t* input, int i, const LowNormLayout& layout) {
    return low_precision_value(input, i, layout.dtype);
}
ELMAN_LOW_NORM_HD inline float low_norm_round(float value, const LowNormLayout& l) {
    if (!l.rounded) return value;
    const uint8_t code = low_precision_encode(value, l.dtype);
    return l.dtype == DType::Float4E2M1FN ? float(numeric::float4_to_double(code)) : numeric::float8_to_float(code, l.dtype);
}
#ifndef __CUDACC__
inline void low_cpu_merge_moments(int added, float mean_added, float variance_added, int& count, float& mean, float& variance) {
    const int combined = count + added;
    const float ratio = combined ? float(added) / float(combined) : 0.f;
    const float difference = mean_added - mean;
    mean += ratio * difference;
    variance += variance_added + difference * difference * ratio * float(count);
    count = combined;
}
inline void low_cpu_row_moments(const uint32_t* input, int row, const LowNormLayout& l, float& mean, float& variance) {
    const int vectors = l.width / 8, chunks = (vectors + 15) / 16;
    int depth = 1;
    while ((int64_t(1) << depth) < chunks) ++depth;
    int counts[32]{};
    float means[32][8]{}, variances[32][8]{};
    for (int chunk = 0; chunk < chunks; ++chunk) {
        const int size = std::min(16, vectors - chunk * 16);
        for (int lane = 0; lane < 8; ++lane) {
            float m = 0, v = 0;
            for (int j = 0; j < size; ++j) {
                const float x = low_norm_value(input, l.index(row, (chunk * 16 + j) * 8 + lane), l);
                const float delta = x - m;
                m += delta * (1.f / float(j + 1)); v += delta * (x - m);
            }
            int n = counts[0]; low_cpu_merge_moments(size, m, v, n, means[0][lane], variances[0][lane]);
        }
        counts[0] += size;
        int mask = chunk + 1;
        for (int level = 1; level < depth && !(mask & 1); ++level, mask >>= 1) {
            for (int lane = 0; lane < 8; ++lane) {
                int n = counts[level];
                low_cpu_merge_moments(counts[level - 1], means[level - 1][lane], variances[level - 1][lane], n, means[level][lane], variances[level][lane]);
                means[level - 1][lane] = variances[level - 1][lane] = 0;
            }
            counts[level] += counts[level - 1]; counts[level - 1] = 0;
        }
    }
    for (int level = 1; level < depth; ++level) {
        for (int lane = 0; lane < 8; ++lane) {
            int n = counts[0]; low_cpu_merge_moments(counts[level], means[level][lane], variances[level][lane], n, means[0][lane], variances[0][lane]);
        }
        counts[0] += counts[level];
    }
    int count = 0; mean = variance = 0;
    for (int j = vectors * 8; j < l.width; ++j) {
        const float x = low_norm_value(input, l.index(row, j), l), delta = x - mean;
        mean += delta / float(++count); variance += delta * (x - mean);
    }
    for (int lane = 0; lane < 8; ++lane) low_cpu_merge_moments(vectors, means[0][lane], variances[0][lane], count, mean, variance);
}
#endif
ELMAN_LOW_NORM_HD inline void low_norm_statistics(const uint32_t* a, float* stats, int row, const LowNormLayout& l) {
    float average = 0, variance = 0;
    const bool rms = l.operation == LowPrecisionNormalization::Rms;
    for (int j = 0; j < l.width; ++j) {
        const float x = low_norm_value(a, l.index(row, j), l);
        if (rms) variance += low_norm_mul(x, x);
        else {
            const float difference = x - average;
            average += difference / float(j + 1);
            variance = low_norm_mad(difference, x - average, variance);
        }
    }
#ifndef __CUDACC__
    if (l.operation == LowPrecisionNormalization::Layer || l.operation == LowPrecisionNormalization::Group)
        low_cpu_row_moments(a, row, l, average, variance);
#endif
#ifndef __CUDACC__
    if (rms) variance = contiguous_float_sum(l.width, [&](int j) { const float x = low_norm_value(a, l.index(row, j), l); return x * x; });
    if (l.operation == LowPrecisionNormalization::Batch) {
        double total = 0, squares = 0;
        for (int j = 0; j < l.width; ++j) total += low_norm_value(a, l.index(row, j), l);
        if (l.spatial == 1) {
            float narrow = 0;
            for (int j = 0; j < l.width; ++j) narrow += low_norm_value(a, l.index(row, j), l);
            total = narrow;
        }
        average = float(total / l.width);
        for (int j = 0; j < l.width; ++j) { const float delta = low_norm_value(a, l.index(row, j), l) - average; squares += delta * delta; }
        if (l.spatial == 1) {
            float narrow = 0;
            for (int j = 0; j < l.width; ++j) { const float delta = low_norm_value(a, l.index(row, j), l) - average; narrow += delta * delta; }
            squares = narrow;
        }
        variance = float(squares);
    }
#endif
    variance /= float(l.width);
    float inverse;
#ifdef __CUDACC__
    inverse = rsqrtf(variance + l.epsilon);
#else
    inverse = (l.operation == LowPrecisionNormalization::Group || l.operation == LowPrecisionNormalization::Batch) ? float(1.0 / ::sqrt(double(variance) + l.wide_epsilon)) : 1.f / ::sqrtf(variance + l.epsilon);
    if (l.operation == LowPrecisionNormalization::Batch && variance == 0 && l.wide_epsilon == 0) inverse = 0;
#endif
    stats[row * 2] = average; stats[row * 2 + 1] = inverse;
}
#ifndef __CUDACC__
inline void low_cpu_batch_dots(const uint32_t* a, const uint32_t* g, const float* stats, int row,
                              const LowNormLayout& l, double& dot, double& total) {
    dot = total = 0;
    if (l.spatial == 1) {
        float d = 0, t = 0;
        for (int j = 0; j < l.width; ++j) {
            const int i = l.index(row, j); const float value = low_norm_value(g, i, l);
            d += (low_norm_value(a, i, l) - stats[row * 2]) * value; t += value;
        }
        dot = d; total = t; return;
    }
    for (int first = 0; first < l.width; first += l.spatial) {
        float products[8]{}, gradients[8]{};
        for (int j = 0; j < l.spatial; ++j) {
            const int i = l.index(row, first + j);
            const float value = low_norm_value(g, i, l), product = (low_norm_value(a, i, l) - stats[row * 2]) * value;
            if (j < 8) { products[j] = product; gradients[j] = value; }
            else { products[j % 8] += product; gradients[j % 8] += value; }
        }
        float ds = products[0], db = gradients[0];
        for (int lane = 1; lane < std::min(8, l.spatial); ++lane) { ds += products[lane]; db += gradients[lane]; }
        dot += ds; total += db;
    }
}
#endif
ELMAN_LOW_NORM_HD inline void low_norm_coefficients(const uint32_t* a, const uint32_t* w, const uint32_t* g,
                                                   const float* stats, float* coefficients, int row, const LowNormLayout& l) {
    float dot = 0, total = 0;
    for (int j = 0; j < l.width; ++j) {
        const int i = l.index(row, j);
        const float weighted = low_norm_mul(low_norm_value(g, i, l), low_norm_value(w, l.weight(row, j), l));
        dot += low_norm_mul(weighted, low_norm_value(a, i, l)); total += weighted;
    }
#ifndef __CUDACC__
    if (l.operation == LowPrecisionNormalization::Layer) {
        float products[8]{}, gradients[8]{};
        for (int j = 0; j < l.width; ++j) {
            const int i = l.index(row, j);
            const float gradient = low_norm_value(g, i, l), value = low_norm_value(a, i, l), weight = low_norm_value(w, j, l);
            const float product = (gradient * value) * weight, weighted = gradient * weight;
            if (j < 8) { products[j] = product; gradients[j] = weighted; }
            else { products[j % 8] += product; gradients[j % 8] += weighted; }
        }
        dot = products[0]; total = gradients[0];
        for (int j = 1; j < std::min(8, l.width); ++j) { dot += products[j]; total += gradients[j]; }
    }
#endif
#ifndef __CUDACC__
    if (l.operation == LowPrecisionNormalization::Rms)
        dot = contiguous_float_sum(l.width, [&](int j) { const int i = l.index(row, j); return low_norm_round(low_norm_value(g, i, l) * low_norm_value(w, j, l), l) * low_norm_value(a, i, l); });
    if (l.operation == LowPrecisionNormalization::Batch) {
        double d, t; low_cpu_batch_dots(a, g, stats, row, l, d, t); dot = float(d); total = float(t / l.width);
    }
#endif
    coefficients[row * 2] = dot; coefficients[row * 2 + 1] = total;
}
ELMAN_LOW_NORM_HD inline float low_norm_result(int i, const uint32_t* a, const uint32_t* b, const uint32_t* c,
                                              const float* stats, const float* coefficients, const LowNormLayout& l) {
    const bool rms = l.operation == LowPrecisionNormalization::Rms;
    if (l.operation == LowPrecisionNormalization::Batch && !l.training) {
        const int channel = i / l.spatial % l.channels;
        const float x = low_norm_value(a, i, l), mean = low_norm_value(b, channel, l), variance = low_norm_value(c, channel, l);
#ifdef __CUDACC__
        return __fadd_rn(__fmul_rn(x - mean, rsqrtf(variance + l.epsilon)), 0.f);
#else
        const float inverse = 1.f / ::sqrtf(variance + l.epsilon);
        return x * inverse + (0.f - mean * inverse);
#endif
    }
    if (l.mode >= 2) {
#ifdef __CUDACC__
        if (l.operation == LowPrecisionNormalization::Batch && l.width)
            return l.mode == 3 ? coefficients[i * 2 + 1] : __fmul_rn(coefficients[i * 2], stats[i * 2 + 1]);
#endif
        if (l.operation == LowPrecisionNormalization::Group && l.width == 0 && l.rows > 0 && l.mode == 2) {
#ifdef __CUDACC__
            return NAN;
#else
            return -std::numeric_limits<float>::quiet_NaN();
#endif
        }
#ifndef __CUDACC__
        if (l.operation == LowPrecisionNormalization::Batch && l.width) {
            double dot, total; low_cpu_batch_dots(a, b, stats, i, l, dot, total);
            return float(l.mode == 3 ? total : dot * stats[i * 2 + 1]);
        }
#endif
        float total = 0;
        for (int row = 0; row < l.rows; ++row) {
            if (l.operation == LowPrecisionNormalization::Batch && row != i) continue;
            if (l.operation == LowPrecisionNormalization::Group && row % l.groups != i / (l.channels / l.groups)) continue;
            const int start = l.operation == LowPrecisionNormalization::Batch ? 0 : l.operation == LowPrecisionNormalization::Group ? (i % (l.channels / l.groups)) * l.spatial : i;
            const int count = l.operation == LowPrecisionNormalization::Batch ? l.width : l.operation == LowPrecisionNormalization::Group ? l.spatial : 1;
            if (l.operation == LowPrecisionNormalization::Group && l.mode == 2) {
#ifdef __CUDACC__
                if (l.spatial == 1) {
                    const int index = l.index(row, start);
                    const float gradient = low_norm_value(b, index, l);
                    const float centered = __fmaf_rn(gradient, low_norm_value(a, index, l), -__fmul_rn(gradient, stats[row * 2]));
                    total = __fmaf_rn(centered, stats[row * 2 + 1], total);
                    continue;
                }
#endif
                float ds = 0, db = 0;
#ifndef __CUDACC__
                float products[8]{}, gradients[8]{};
#endif
                for (int j = 0; j < count; ++j) {
                    const int index = l.index(row, start + j);
                    const float upstream = low_norm_value(b, index, l), value = low_norm_value(a, index, l);
#ifdef __CUDACC__
                    ds = low_norm_mad(upstream, value, ds); db += upstream;
#else
                    if (j < 8) { products[j] = upstream * value; gradients[j] = upstream; }
                    else { products[j % 8] += upstream * value; gradients[j % 8] += upstream; }
#endif
                }
#ifndef __CUDACC__
                ds = products[0]; db = gradients[0];
                for (int j = 1; j < std::min(8, count); ++j) { ds += products[j]; db += gradients[j]; }
#endif
                total = low_norm_mad(low_norm_mad(-db, stats[row * 2], ds), stats[row * 2 + 1], total);
                continue;
            }
            for (int j = start; j < start + count && j < l.width; ++j) {
                const int index = l.index(row, j);
                const float upstream = low_norm_value(b, index, l);
                float normalized = low_norm_mul(low_norm_value(a, index, l) - stats[row * 2], stats[row * 2 + 1]);
#ifndef __CUDACC__
                if (l.operation == LowPrecisionNormalization::Layer)
                    normalized = low_norm_value(a, index, l) * stats[row * 2 + 1] + (-stats[row * 2 + 1] * stats[row * 2]);
#endif
#ifdef __CUDACC__
                if (l.operation == LowPrecisionNormalization::Layer && l.mode == 2) {
                    total = __fmaf_rn(upstream * (low_norm_value(a, index, l) - stats[row * 2]), stats[row * 2 + 1], total);
                    continue;
                }
#endif
                const float term = l.mode == 3 ? upstream : low_norm_round(low_norm_mul(upstream, low_norm_round(normalized, l)), l);
                if (rms && l.unbroadcasted) return term;
                total += term;
            }
        }
        return total;
    }
    const int row = l.operation == LowPrecisionNormalization::Batch ? i / l.spatial % l.channels : i / l.width;
    const int j = l.operation == LowPrecisionNormalization::Batch ? i / (l.spatial * l.channels) * l.spatial + i % l.spatial : i % l.width;
    const float x = low_norm_value(a, i, l), weight = l.has_weight ? low_norm_value(b, l.weight(row, j), l) : 1.f;
    const float average = stats[row * 2], inverse = stats[row * 2 + 1];
    if (l.mode == 0) {
        if (rms) {
            float value = low_norm_round(low_norm_mul(x, inverse), l);
            if (l.has_weight) value = low_norm_mul(value, weight);
            return l.has_bias ? value + low_norm_value(c, l.weight(row, j), l) : value;
        }
        const float bias = l.has_bias ? low_norm_value(c, l.weight(row, j), l) : 0.f;
        if (l.operation == LowPrecisionNormalization::Group && !l.has_weight && !l.has_bias) return low_norm_mul(x - average, inverse);
        if (l.operation == LowPrecisionNormalization::Layer) {
            const float normalized = low_norm_mul(x - average, inverse);
#ifdef __CUDACC__
            if (l.width % 4 != 0 || l.width > (1 << 24)) return low_norm_mad(normalized, weight, bias);
            if (!l.has_bias) return l.has_weight ? low_norm_mul(normalized, weight) : normalized;
            return l.has_weight ? low_norm_mad(normalized, weight, bias) : normalized + bias;
#else
            return low_norm_mad(normalized, weight, bias);
#endif
        }
#ifdef __CUDACC__
        if (l.operation == LowPrecisionNormalization::Group && l.spatial == 1) {
            const float normalized = low_norm_mul(x - average, inverse);
            if (!l.has_bias) return low_norm_mul(normalized, weight);
            return l.has_weight ? low_norm_mad(normalized, weight, bias) : normalized + bias;
        }
        if (l.operation == LowPrecisionNormalization::Batch)
            return low_norm_mad(low_norm_mul(weight, x - average), inverse, bias);
#endif
        const float scale = low_norm_mul(inverse, weight);
#ifdef __CUDACC__
        if (l.operation == LowPrecisionNormalization::Group)
            // The separately stored affine offset rounds its product before adding beta.
            return low_norm_mad(scale, x, __fadd_rn(-__fmul_rn(scale, average), bias));
#endif
        return low_norm_mad(scale, x, low_norm_mad(-scale, average, bias));
    }
    const float upstream = low_norm_value(c, i, l), weighted = low_norm_round(low_norm_mul(upstream, weight), l);
    const float dot = coefficients[row * 2], total = coefficients[row * 2 + 1];
#ifdef __CUDACC__
    if (l.operation == LowPrecisionNormalization::Batch) {
        const float reciprocal = 1.f / float(l.width), gradient_mean = __fmul_rn(total, reciprocal);
        const float scale = inverse * weight;
        if (l.spatial == 1) {
            const float projection = inverse * inverse * dot * reciprocal;
            const float centered_gradient = __fmaf_rn(-total, reciprocal, upstream);
            return __fmul_rn(__fmaf_rn(-(x - average), projection, centered_gradient), scale);
        }
        const float projection = dot * reciprocal * inverse * inverse;
        return __fmul_rn(__fsub_rn(__fmaf_rn(-(x - average), projection, upstream), gradient_mean), scale);
    }
    if (l.operation == LowPrecisionNormalization::Layer) {
        const float term = __fmul_rn(1.f / float(l.width), inverse);
        const float projected = __fmul_rn(__fmul_rn(x - average, inverse), dot);
        return __fmul_rn(__fmaf_rn(__fmul_rn(float(l.width), weight), upstream, -projected) - total, term);
    }
#endif
    if (rms) {
#ifdef __CUDACC__
        // ATen's CUDA scalar exponent-three specialization uses two rounded multiplies.
        const float cube = __fmul_rn(__fmul_rn(inverse, inverse), inverse);
#else
        const float cube = ::powf(inverse, 3.f);
#endif
        const float variance_gradient = low_norm_mul(low_norm_mul(dot, -.5f), cube);
#ifdef __CUDACC__
        const float square_gradient = __fmul_rn(variance_gradient, 1.f / float(l.width));
#else
        const float square_gradient = variance_gradient / float(l.width);
#endif
        return low_norm_mul(weighted, inverse) + low_norm_mul(square_gradient, low_norm_mul(x, 2.f));
    }
#ifndef __CUDACC__
    if (l.operation == LowPrecisionNormalization::Batch) {
        const float k = dot * inverse * inverse / float(l.width);
        return ((upstream - total - (x - average) * k) * inverse) * weight;
    }
    if (l.operation == LowPrecisionNormalization::Layer || l.operation == LowPrecisionNormalization::Group) {
        const float scale = 1.f / float(l.width);
        const float c2 = (total * average - dot) * inverse * inverse * inverse * scale;
        const float c3 = -c2 * average - total * inverse * scale;
        const float first = l.operation == LowPrecisionNormalization::Group ? (inverse * weight) * upstream : (inverse * upstream) * weight;
        const float result = first + c2 * x + c3;
        // AVX2 invalid arithmetic produces the negative indefinite NaN.
        return std::isnan(result) ? -std::numeric_limits<float>::quiet_NaN() : result;
    }
#endif
    const float c2 = low_norm_mul(low_norm_mul(low_norm_mul(low_norm_mad(total, average, -dot), inverse), inverse), inverse) * (1.f / float(l.width));
    const float c3 = low_norm_mad(-c2, average, -low_norm_mul(low_norm_mul(total, inverse), 1.f / float(l.width)));
    return low_norm_mad(low_norm_mul(weight, inverse), upstream, low_norm_mul(c2, x)) + c3;
}
} // namespace elman
