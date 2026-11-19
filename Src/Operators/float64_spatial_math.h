#pragma once
#include "Operators/float64_spatial.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_SPATIAL_HD __host__ __device__
#else
#define ELMAN_SPATIAL_HD
#endif
namespace elman {
struct Float64Image {
    int n, c, h, w;
    ELMAN_SPATIAL_HD int index(int batch, int channel, int y, int x) const {
        return ((batch * c + channel) * h + y) * w + x;
    }
};
ELMAN_SPATIAL_HD inline Float64Image float64_image(const KernelParams& p, int input) {
    return input < 0 ? Float64Image{p.output_shape(0), p.output_shape(1), p.output_shape(2), p.output_shape(3)} :
        Float64Image{p.input_shape(KernelInput(input), 0), p.input_shape(KernelInput(input), 1),
                     p.input_shape(KernelInput(input), 2), p.input_shape(KernelInput(input), 3)};
}
ELMAN_SPATIAL_HD inline double float64_convolution_value(Float64Spatial op, int i, const double* a,
                                                         const double* b, const double* bias, const KernelParams& p) {
    const bool forward = op == Float64Spatial::Convolution, input_gradient = op == Float64Spatial::InputGradient;
    const auto x = float64_image(p, input_gradient ? -1 : 0);
    const auto w = float64_image(p, op == Float64Spatial::WeightGradient ? -1 : 1);
    const auto y = float64_image(p, forward ? -1 : input_gradient ? 0 : 1);
    const int sh = p.attribute(0), sw = p.attribute(1), ph = p.attribute(2), pw = p.attribute(3);
    const int dh = p.attribute(4), dw = p.attribute(5), outputs_per_group = w.n / p.attribute(6);
    double value = 0.;
    if (forward) {
        const int ox = i % y.w, oy = i / y.w % y.h, oc = i / (y.w * y.h) % y.c, batch = i / (y.c * y.h * y.w);
        if (p.input_rank(KernelInput::C) >= 0) value = bias[oc];
        const int first_channel = oc / outputs_per_group * w.c;
        for (int ic = 0; ic < w.c; ++ic) for (int ky = 0; ky < w.h; ++ky) for (int kx = 0; kx < w.w; ++kx) {
            const int64_t iy = int64_t(oy) * sh - ph + int64_t(ky) * dh;
            const int64_t ix = int64_t(ox) * sw - pw + int64_t(kx) * dw;
#if defined(__CUDA_ARCH__)
            // cuDNN excludes padding from nonfinite products; the CPU im2col
            // reference instead multiplies explicit padding zeros.
            if (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) continue;
#endif
            const double input = iy >= 0 && iy < x.h && ix >= 0 && ix < x.w ? a[x.index(batch, first_channel + ic, int(iy), int(ix))] : 0.;
            // Explicit padding participates in nonfinite products just as an
            // im2col/GEMM convolution does.
            value = ::fma(input, b[w.index(oc, ic, ky, kx)], value);
        }
    } else if (input_gradient) {
        const int ix = i % x.w, iy = i / x.w % x.h, ic = i / (x.w * x.h) % x.c, batch = i / (x.c * x.h * x.w);
        const int first_output = ic / w.c * outputs_per_group, local_channel = ic % w.c;
        for (int local_output = 0; local_output < outputs_per_group; ++local_output)
            for (int ky = 0; ky < w.h; ++ky) for (int kx = 0; kx < w.w; ++kx) {
                const int64_t ny = int64_t(iy) + ph - int64_t(ky) * dh, nx = int64_t(ix) + pw - int64_t(kx) * dw;
                if (ny < 0 || nx < 0 || ny % sh || nx % sw || ny / sh >= y.h || nx / sw >= y.w) continue;
                const int oc = first_output + local_output;
                value = ::fma(a[y.index(batch, oc, int(ny / sh), int(nx / sw))], b[w.index(oc, local_channel, ky, kx)], value);
            }
    } else {
        const int kx = i % w.w, ky = i / w.w % w.h, local_channel = i / (w.w * w.h) % w.c, oc = i / (w.c * w.h * w.w);
        const int ic = oc / outputs_per_group * w.c + local_channel;
        for (int batch = 0; batch < x.n; ++batch) for (int oy = 0; oy < y.h; ++oy) for (int ox = 0; ox < y.w; ++ox) {
            const int64_t iy = int64_t(oy) * sh - ph + int64_t(ky) * dh, ix = int64_t(ox) * sw - pw + int64_t(kx) * dw;
#if defined(__CUDA_ARCH__)
            if (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) continue;
#endif
            const double input = iy >= 0 && iy < x.h && ix >= 0 && ix < x.w ? a[x.index(batch, ic, int(iy), int(ix))] : 0.;
            value = ::fma(input, b[y.index(batch, oc, oy, ox)], value);
        }
    }
    return value;
}
struct Float64PoolWindow { double sum, maximum; int selected; int64_t count; };
ELMAN_SPATIAL_HD inline Float64PoolWindow float64_pool_window(const double* a, Float64Image x, int plane,
                                                              int oy, int ox, const KernelParams& p) {
    Float64PoolWindow result{0., -HUGE_VAL, -1, 0};
    const int64_t start_y = int64_t(oy) * p.attribute(2) - p.attribute(4);
    const int64_t start_x = int64_t(ox) * p.attribute(3) - p.attribute(5);
    for (int ky = 0; ky < p.attribute(0); ++ky) for (int kx = 0; kx < p.attribute(1); ++kx) {
        const int64_t iy = start_y + ky, ix = start_x + kx;
        if (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) continue;
        const int index = (plane * x.h + int(iy)) * x.w + int(ix);
        const double value = a[index];
        result.sum += value;
        ++result.count;
        if (result.selected < 0 || value > result.maximum || value != value) {
            result.maximum = value; result.selected = index;
        }
    }
    if (p.attribute(6)) result.count = int64_t(p.attribute(0)) * p.attribute(1);
    return result;
}
ELMAN_SPATIAL_HD inline double float64_spatial_value(Float64Spatial op, int i, const double* a,
                                                     const double* b, const double* c, const KernelParams& p) {
    if (op <= Float64Spatial::WeightGradient) return float64_convolution_value(op, i, a, b, c, p);
    const bool gradient = op == Float64Spatial::MaximumPoolGradient || op == Float64Spatial::AveragePoolGradient;
    const bool maximum = op == Float64Spatial::MaximumPool || op == Float64Spatial::MaximumPoolGradient;
    const auto x = float64_image(p, 0), y = float64_image(p, gradient ? 1 : -1);
    if (!gradient) {
        const auto window = float64_pool_window(a, x, i / (y.h * y.w), i / y.w % y.h, i % y.w, p);
        return maximum ? window.maximum : window.sum / double(window.count);
    }
    const int ix = i % x.w, iy = i / x.w % x.h, plane = i / (x.h * x.w);
    double value = 0.;
    for (int oy = 0; oy < y.h; ++oy) for (int ox = 0; ox < y.w; ++ox) {
        const int64_t sy = int64_t(oy) * p.attribute(2) - p.attribute(4), sx = int64_t(ox) * p.attribute(3) - p.attribute(5);
        if (iy < sy || iy >= sy + p.attribute(0) || ix < sx || ix >= sx + p.attribute(1)) continue;
        const auto window = float64_pool_window(a, x, plane, oy, ox, p);
        const double upstream = b[(plane * y.h + oy) * y.w + ox];
        if (maximum) { if (window.selected == i) value += upstream; }
        else value += upstream / double(window.count);
    }
    return value;
}
} // namespace elman
#undef ELMAN_SPATIAL_HD
