#pragma once
#include "Operators/low_precision_spatial.h"
#include <cmath>
#include "Operators/low_precision_pointwise_math.h"

#ifdef __CUDACC__
#define ELMAN_SPATIAL_HD __device__
#else
#define ELMAN_SPATIAL_HD
#endif
namespace elman {
struct LowSpatialValues {
    const uint32_t* data;
    DType dtype;
    ELMAN_SPATIAL_HD float operator[](int i) const { return low_precision_value(data, i, dtype); }
};
struct LowSpatialImage {
    int n, c, h, w;
    ELMAN_SPATIAL_HD int index(int batch, int channel, int y, int x) const {
        return ((batch * c + channel) * h + y) * w + x;
    }
};
ELMAN_SPATIAL_HD inline LowSpatialImage low_spatial_image(const KernelParams& p, int input) {
    return input < 0 ? LowSpatialImage{p.output_shape(0), p.output_shape(1), p.output_shape(2), p.output_shape(3)} :
        LowSpatialImage{p.input_shape(KernelInput(input), 0), p.input_shape(KernelInput(input), 1),
                     p.input_shape(KernelInput(input), 2), p.input_shape(KernelInput(input), 3)};
}
ELMAN_SPATIAL_HD inline float low_convolution_value(LowPrecisionSpatial op, int i, LowSpatialValues a,
                                                         LowSpatialValues b, LowSpatialValues bias, const KernelParams& p) {
    const bool forward = op == LowPrecisionSpatial::Convolution, input_gradient = op == LowPrecisionSpatial::InputGradient;
    const LowSpatialImage x = low_spatial_image(p, input_gradient ? -1 : 0);
    const LowSpatialImage w = low_spatial_image(p, op == LowPrecisionSpatial::WeightGradient ? -1 : 1);
    const LowSpatialImage y = low_spatial_image(p, forward ? -1 : input_gradient ? 0 : 1);
    const int sh = p.attribute(0), sw = p.attribute(1), ph = p.attribute(2), pw = p.attribute(3);
    const int dh = p.attribute(4), dw = p.attribute(5), outputs_per_group = w.n / p.attribute(6);
#ifndef __CUDACC__
    // The direct convolution path excludes padding from nonfinite products.
    const bool direct = (sh != 1 || sw != 1 || dh != 1 || dw != 1 || x.n >= 16 || w.h != 1 || w.w != 1) &&
        (p.attribute(6) > 1 || (w.h > 3 && w.w > 3) || x.n > 1 || int64_t(x.n) * x.c * x.h * x.w > 20480);
#else
    const bool direct = p.attribute(6) > 1 && x.c == p.attribute(6);
#endif
    float value = 0.;
    if (forward) {
        const int ox = i % y.w, oy = i / y.w % y.h, oc = i / (y.w * y.h) % y.c, batch = i / (y.c * y.h * y.w);
        const int first_channel = oc / outputs_per_group * w.c;
        for (int ic = 0; ic < w.c; ++ic) for (int ky = 0; ky < w.h; ++ky) for (int kx = 0; kx < w.w; ++kx) {
            const int64_t iy = int64_t(oy) * sh - ph + int64_t(ky) * dh;
            const int64_t ix = int64_t(ox) * sw - pw + int64_t(kx) * dw;
            if (direct && (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w)) continue;

            const float input = iy >= 0 && iy < x.h && ix >= 0 && ix < x.w ? a[x.index(batch, first_channel + ic, int(iy), int(ix))] : 0.;
            // Explicit padding participates in nonfinite products just as an
            // im2col/GEMM convolution does.
            const float weight = b[w.index(oc, ic, ky, kx)];
#ifndef __CUDACC__
            // The CPU single-product GEMM preserves the product's signed zero.
            if (w.c == 1 && w.h == 1 && w.w == 1) value = input * weight;
            else
#else
            // With a bias, CUDA's single-product path adds the product to beta*C.
            if (w.c == 1 && w.h == 1 && w.w == 1 && p.input_rank(KernelInput::C) >= 0) value = __fmul_rn(input, weight);
            else
#endif
                value = ::fmaf(input, weight, value);
        }
        if (p.input_rank(KernelInput::C) >= 0) value += bias[oc];
    } else if (input_gradient) {
        const int ix = i % x.w, iy = i / x.w % x.h, ic = i / (x.w * x.h) % x.c, batch = i / (x.c * x.h * x.w);
        const int first_output = ic / w.c * outputs_per_group, local_channel = ic % w.c;
        for (int local_output = 0; local_output < outputs_per_group; ++local_output)
            for (int ky = 0; ky < w.h; ++ky) for (int kx = 0; kx < w.w; ++kx) {
                const int64_t ny = int64_t(iy) + ph - int64_t(ky) * dh, nx = int64_t(ix) + pw - int64_t(kx) * dw;
                if (ny < 0 || nx < 0 || ny % sh || nx % sw || ny / sh >= y.h || nx / sw >= y.w) continue;
                const int oc = first_output + local_output;
                value = ::fmaf(a[y.index(batch, oc, int(ny / sh), int(nx / sw))], b[w.index(oc, local_channel, ky, kx)], value);
            }
    } else {
        const int kx = i % w.w, ky = i / w.w % w.h, local_channel = i / (w.w * w.h) % w.c, oc = i / (w.c * w.h * w.w);
        const int ic = oc / outputs_per_group * w.c + local_channel;
        for (int batch = 0; batch < x.n; ++batch) for (int oy = 0; oy < y.h; ++oy) for (int ox = 0; ox < y.w; ++ox) {
            const int64_t iy = int64_t(oy) * sh - ph + int64_t(ky) * dh, ix = int64_t(ox) * sw - pw + int64_t(kx) * dw;
            if (direct && (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w)) continue;

            const float input = iy >= 0 && iy < x.h && ix >= 0 && ix < x.w ? a[x.index(batch, ic, int(iy), int(ix))] : 0.;
            value = ::fmaf(input, b[y.index(batch, oc, oy, ox)], value);
        }
    }
    return value;
}
struct LowPoolWindow { float sum, maximum; int selected; int64_t count; };
ELMAN_SPATIAL_HD inline LowPoolWindow low_pool_window(LowSpatialValues a, LowSpatialImage x, int plane,
                                                              int oy, int ox, const KernelParams& p) {
    LowPoolWindow result{0., -HUGE_VALF, -1, 0};
    const int64_t start_y = int64_t(oy) * p.attribute(2) - p.attribute(4);
    const int64_t start_x = int64_t(ox) * p.attribute(3) - p.attribute(5);
    for (int ky = 0; ky < p.attribute(0); ++ky) for (int kx = 0; kx < p.attribute(1); ++kx) {
        const int64_t iy = start_y + ky, ix = start_x + kx;
        if (iy < 0 || iy >= x.h || ix < 0 || ix >= x.w) continue;
        const int index = (plane * x.h + int(iy)) * x.w + int(ix);
        const float value = a[index];
        result.sum += value;
        ++result.count;
        if (result.selected < 0 || value > result.maximum || value != value) {
            result.maximum = value; result.selected = index;
        }
    }
    if (p.attribute(6)) result.count = int64_t(p.attribute(0)) * p.attribute(1);
    return result;
}
ELMAN_SPATIAL_HD inline float low_precision_spatial_value(LowPrecisionSpatial op, int i, LowSpatialValues a,
                                                     LowSpatialValues b, LowSpatialValues c, const KernelParams& p) {
    if (op <= LowPrecisionSpatial::WeightGradient) return low_convolution_value(op, i, a, b, c, p);
    const bool gradient = op == LowPrecisionSpatial::MaximumPoolGradient || op == LowPrecisionSpatial::AveragePoolGradient;
    const bool maximum = op == LowPrecisionSpatial::MaximumPool || op == LowPrecisionSpatial::MaximumPoolGradient;
    const LowSpatialImage x = low_spatial_image(p, 0), y = low_spatial_image(p, gradient ? 1 : -1);
    if (!gradient) {
        const LowPoolWindow window = low_pool_window(a, x, i / (y.h * y.w), i / y.w % y.h, i % y.w, p);
        return maximum ? window.maximum : window.sum / float(window.count);
    }
    const int ix = i % x.w, iy = i / x.w % x.h, plane = i / (x.h * x.w);
    float value = 0.;
    for (int oy = 0; oy < y.h; ++oy) for (int ox = 0; ox < y.w; ++ox) {
        const int64_t sy = int64_t(oy) * p.attribute(2) - p.attribute(4), sx = int64_t(ox) * p.attribute(3) - p.attribute(5);
        if (iy < sy || iy >= sy + p.attribute(0) || ix < sx || ix >= sx + p.attribute(1)) continue;
        const LowPoolWindow window = low_pool_window(a, x, plane, oy, ox, p);
        const float upstream = b[(plane * y.h + oy) * y.w + ox];
        if (maximum) { if (window.selected == i) value += upstream; }
        else value += upstream / float(window.count);
    }
    return value;
}
} // namespace elman
#undef ELMAN_SPATIAL_HD
