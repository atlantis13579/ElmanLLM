#pragma once
#include "Operators/float64_normalization.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_DOUBLE_NORM_HD __host__ __device__
#else
#define ELMAN_DOUBLE_NORM_HD
#endif
namespace elman {
// Slots 5-7 extend epsilon without changing legacy attributes or RMS rounding.
ELMAN_DOUBLE_NORM_HD inline double float64_normalization_epsilon(Float64Normalization op, const KernelParams& p) {
    const bool group = op == Float64Normalization::Group || op == Float64Normalization::GroupInputGradient || op == Float64Normalization::GroupWeightGradient;
    if (p.attribute(7) || (group && p.attribute(2))) {
        const int slot = p.attribute(7) ? 5 : 3;
        union { uint64_t bits; double value; } scalar;
        scalar.bits = uint64_t(uint32_t(p.attribute(slot))) | (uint64_t(uint32_t(p.attribute(slot + 1))) << 32);
        return scalar.value;
    }
    union { uint32_t bits; float value; } scalar;
    scalar.bits = uint32_t(p.attribute(op == Float64Normalization::Batch ? 0 : 1));
    return double(scalar.value);
}
ELMAN_DOUBLE_NORM_HD inline double float64_norm_multiply_add(double a, double b, double c) {
#if defined(__CUDA_ARCH__)
    return ::fma(a, b, c);
#else
    return a * b + c;
#endif
}
ELMAN_DOUBLE_NORM_HD inline double float64_norm_multiply(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dmul_rn(a, b);
#else
    return a * b;
#endif
}
ELMAN_DOUBLE_NORM_HD inline double float64_norm_add(double a, double b) {
#if defined(__CUDA_ARCH__)
    return __dadd_rn(a, b);
#else
    return a + b;
#endif
}
ELMAN_DOUBLE_NORM_HD inline void float64_norm_statistics(const double* input, int base, int length, double epsilon, bool rms, double& mean, double& inverse) {
    mean = 0.;
    double variance = 0.;
    if (rms) {
        for (int j = 0; j < length; ++j)
            variance = float64_norm_add(variance, float64_norm_multiply(input[base + j], input[base + j]));
    } else {
        // Welford keeps constant large rows centered exactly and avoids losing
        // their small variance when accumulating the mean.
        for (int j = 0; j < length; ++j) {
            const double value = input[base + j], delta = value - mean;
            mean += delta / double(j + 1);
            variance = float64_norm_add(variance, float64_norm_multiply(delta, value - mean));
        }
    }
    inverse = 1. / ::sqrt(variance / double(length) + epsilon);
}
// Preserve the reference's per-channel reduction order, including the CPU
// four-Double lanes and CUDA warp reduction. Cancellation in affine gradients
// can otherwise magnify the last bit of a large constant input.
ELMAN_DOUBLE_NORM_HD inline void float64_norm_channel_gradients(const double* input, const double* upstream,
        int base, int length, double& product, double& gradient) {
#if defined(__CUDA_ARCH__)
    const int threads = length < 512 ? 32 : 512;
    double warp_product[16] = {}, warp_gradient[16] = {};
    for (int warp = 0; warp < threads / 32; ++warp) {
        double products[32] = {}, gradients[32] = {};
        for (int lane = 0; lane < 32; ++lane)
            for (int j = warp * 32 + lane; j < length; j += threads) {
                products[lane] = ::fma(upstream[base + j], input[base + j], products[lane]);
                gradients[lane] += upstream[base + j];
            }
        for (int offset = 16; offset > 0; offset /= 2)
            for (int lane = 0; lane < offset; ++lane) {
                products[lane] += products[lane + offset];
                gradients[lane] += gradients[lane + offset];
            }
        warp_product[warp] = products[0]; warp_gradient[warp] = gradients[0];
    }
    for (int offset = 8; offset > 0; offset /= 2)
        for (int lane = 0; lane < offset; ++lane) {
            warp_product[lane] += warp_product[lane + offset];
            warp_gradient[lane] += warp_gradient[lane + offset];
        }
    product = warp_product[0]; gradient = warp_gradient[0];
#else
    double products[4] = {}, gradients[4] = {};
    for (int j = 0; j < length; ++j) {
        products[j % 4] += upstream[base + j] * input[base + j];
        gradients[j % 4] += upstream[base + j];
    }
    product = products[0]; gradient = gradients[0];
    for (int lane = 1; lane < 4 && lane < length; ++lane) {
        product += products[lane]; gradient += gradients[lane];
    }
#endif
}
ELMAN_DOUBLE_NORM_HD inline int float64_norm_length(Float64Normalization op, const KernelParams& p) {
    if (op == Float64Normalization::Layer || op == Float64Normalization::Rms) {
        int length = 1;
        for (int d = p.input_rank(KernelInput::A) - p.attribute(0); d < p.input_rank(KernelInput::A); ++d)
            length *= p.input_shape(KernelInput::A, d);
        return length;
    }
    return (p.input_shape(KernelInput::A, 1) / p.attribute(0)) * p.input_stride(KernelInput::A, 1);
}
// Each row owns its complete output. Statistics never use Float32 storage.
ELMAN_DOUBLE_NORM_HD inline void float64_normalization_row(Float64Normalization op, int row, double* output,
        const double* a, const double* b, const double* c, const KernelParams& p) {
    const double epsilon = float64_normalization_epsilon(op, p);
    if (op == Float64Normalization::Batch) {
        const int channels = p.input_shape(KernelInput::A, 1), spatial = p.input_stride(KernelInput::A, 1);
        const int channel = row % channels, base = row * spatial;
        const double inverse = 1. / ::sqrt(c[channel] + epsilon);
        for (int j = 0; j < spatial; ++j) output[base + j] = (a[base + j] - b[channel]) * inverse;
        return;
    }
    const bool layer = op == Float64Normalization::Layer || op == Float64Normalization::Rms;
    const int length = float64_norm_length(op, p);
    if (op == Float64Normalization::GroupWeightGradient) {
        const int batch = p.input_shape(KernelInput::A, 0), channels = p.input_shape(KernelInput::A, 1);
        const int spatial = p.input_stride(KernelInput::A, 1), width = channels / p.attribute(0);
        if (!spatial) { output[row] = batch ? NAN : 0.; return; }
        double result = 0.;
        for (int n = 0; n < batch; ++n) {
            double mean, inverse;
            float64_norm_statistics(a, (n * channels + row / width * width) * spatial, length, epsilon, false, mean, inverse);
            double product, gradient;
            const int start = (n * channels + row) * spatial;
            float64_norm_channel_gradients(a, b, start, spatial, product, gradient);
            double centered = float64_norm_multiply_add(-gradient, mean, product);
#if defined(__CUDA_ARCH__)
            if (spatial == 1) centered = ::fma(b[start], a[start], -float64_norm_multiply(b[start], mean));
#endif
            result = float64_norm_multiply_add(centered, inverse, result);
        }
        output[row] = result;
        return;
    }
    const int base = row * length;
    double mean, inverse;
    float64_norm_statistics(a, base, length, epsilon, op == Float64Normalization::Rms, mean, inverse);
    const int channels = layer ? 0 : p.input_shape(KernelInput::A, 1);
    const int spatial = layer ? 1 : p.input_stride(KernelInput::A, 1);
    const int width = layer ? 0 : channels / p.attribute(0);
    if (op == Float64Normalization::GroupInputGradient) {
        double ds = 0., db = 0.;
        for (int channel = 0; channel < width; ++channel) {
            double product, gradient;
            float64_norm_channel_gradients(a, c, base + channel * spatial, spatial, product, gradient);
            const double weight = b[(row * width + channel) % channels];
            ds = float64_norm_add(ds, float64_norm_multiply(product, weight));
            db = float64_norm_add(db, float64_norm_multiply(gradient, weight));
        }
        const double reciprocal = 1. / double(length);
        const double c2 = float64_norm_multiply_add(db, mean, -ds) * inverse * inverse * inverse * reciprocal;
        const double c3 = float64_norm_multiply_add(-c2, mean, -db * inverse * reciprocal);
        for (int j = 0; j < length; ++j) {
            const int i = base + j;
            const double c1 = inverse * b[(i / spatial) % channels];
            output[i] = float64_norm_add(float64_norm_multiply_add(c1, c[i], float64_norm_multiply(c2, a[i])), c3);
        }
        return;
    }
    for (int j = 0; j < length; ++j) {
        const int i = base + j, affine = layer ? j : (i / spatial) % channels;
        const double weight = p.input_rank(KernelInput::B) >= 0 ? b[affine] : 1.;
        const double bias = p.input_rank(KernelInput::C) >= 0 ? c[affine] : 0.;
        bool folded = op == Float64Normalization::Group;
#if defined(__CUDA_ARCH__)
        folded = folded && spatial != 1;
#else
        folded = folded && (p.input_rank(KernelInput::B) >= 0 || p.input_rank(KernelInput::C) >= 0);
#endif
        if (folded) {
            const double scale = inverse * weight;
            output[i] = float64_norm_multiply_add(scale, a[i], float64_norm_multiply_add(-scale, mean, bias));
        } else {
            double value = float64_norm_multiply(a[i] - mean, inverse);
            if (p.input_rank(KernelInput::B) >= 0) value = float64_norm_multiply(value, weight);
            if (p.input_rank(KernelInput::C) >= 0) value = float64_norm_add(value, bias);
            output[i] = value;
        }
    }
}
inline int float64_normalization_rows(Float64Normalization op, const KernelParams& p) {
    if (op == Float64Normalization::GroupWeightGradient) return p.output_count();
    if (!p.output_count()) return 0;
    if (op == Float64Normalization::Batch) return p.output_count() / p.input_stride(KernelInput::A, 1);
    return p.output_count() / float64_norm_length(op, p);
}
} // namespace elman
#undef ELMAN_DOUBLE_NORM_HD
