#pragma once

#include "Operators/affine_quantization.h"
#include <cmath>

#ifdef __CUDACC__
#define ELMAN_AFFINE_HD __device__
#else
#define ELMAN_AFFINE_HD
#endif

namespace elman {
ELMAN_AFFINE_HD inline int64_t affine_zero(const uint32_t* values, int index, int dtype) {
    return dtype == int(DType::Int64) ? reinterpret_cast<const int64_t*>(values)[index] :
                                      reinterpret_cast<const int32_t*>(values)[index];
}
template<class Real>
ELMAN_AFFINE_HD int64_t affine_encode(Real value, Real scale, int64_t zero, DType dtype) {
    const int64_t low = quantized_minimum(dtype), high = quantized_maximum(dtype);
    // Invalid raw metadata has a deterministic result. The public wrapper
    // rejects it before submission. Nonfinite inputs saturate; NaN maps to low.
    if (!(scale > Real(0)) || double(scale) > 1.7976931348623157e308 || zero < low || zero > high) return low;
#if defined(__CUDA_ARCH__)
    const Real inverse = sizeof(Real) == sizeof(float) ? Real(__fdiv_rn(1.0f, float(scale))) : Real(__ddiv_rn(1.0, double(scale)));
#else
    const Real inverse = Real(1) / scale;
#endif
    const Real scaled = value * inverse;
    if (scaled != scaled || double(scaled) <= double(low - zero)) return low;
    if (double(scaled) >= double(high - zero)) return high;
    const double lower = ::floor(double(scaled));
    const double fraction = double(scaled) - lower;
    const int64_t integer = int64_t(lower);
    const int64_t rounded = integer + (fraction > .5 || (fraction == .5 && (integer & 1)));
    const int64_t result = rounded + zero;
    return result < low ? low : result > high ? high : result;
}
template<class Reader>
ELMAN_AFFINE_HD int64_t affine_encode_element(int index, AffineQuantization operation, const uint32_t* input, const uint32_t* scales,
                                             const uint32_t* zeros, const KernelParams& p, Reader read) {
    const int rank = p.input_rank(KernelInput::A);
    const int columns = rank ? p.input_shape(KernelInput::A, rank - 1) : 1;
    const int row = index / columns, group = (index % columns) / p.attribute(0);
    const int scale_index = quantization_parameter_index(KernelInput::B, row, group, p);
    const int zero_index = quantization_parameter_index(KernelInput::C, row, group, p);
    const auto zero = affine_zero(zeros, zero_index, p.input_dtype(KernelInput::C));
    if (operation == AffineQuantization::Relu) {
        const auto dtype = DType(p.output_dtype());
        if (zero < quantized_minimum(dtype) || zero > quantized_maximum(dtype)) return quantized_minimum(dtype);
        const auto value = quantized_code(input, uint64_t(index), DType(p.output_dtype()));
        return value < zero ? zero : value;
    }
    const double value = read(input, index, p.input_dtype(KernelInput::A));
    const double scale = read(scales, scale_index, p.input_dtype(KernelInput::B));
    if (p.input_dtype(KernelInput::A) == int(DType::Float64) || p.input_dtype(KernelInput::B) == int(DType::Float64))
        return affine_encode(value, scale, zero, DType(p.output_dtype()));
    return affine_encode(float(value), float(scale), zero, DType(p.output_dtype()));
}
template<class Reader>
ELMAN_AFFINE_HD double affine_decode_element(int index, AffineQuantization operation,
                                              const uint32_t* input, const uint32_t* scales,
                                              const uint32_t* zeros, const uint32_t* indices,
                                              const KernelParams& p, Reader read) {
    const int rank = p.input_rank(KernelInput::A);
    const int columns = rank ? p.input_shape(KernelInput::A, rank - 1) : 1;
    int row = index / columns;
    const int column = index % columns;
    if (operation == AffineQuantization::Embedding) {
        const int64_t selected = affine_zero(indices, row, p.input_dtype(KernelInput::D));
        // Raw dispatch must remain memory-safe even without the public check.
        if (selected < 0 || selected >= p.input_shape(KernelInput::A, 0)) return 0;
        row = int(selected);
    }
    const int group = column / p.attribute(0);
    const int scale_index = quantization_parameter_index(KernelInput::B, row, group, p);
    const int zero_index = quantization_parameter_index(KernelInput::C, row, group, p);
    const auto dtype = DType(p.input_dtype(KernelInput::A));
    const int64_t zero = affine_zero(zeros, zero_index, p.input_dtype(KernelInput::C));
    const double scale = read(scales, scale_index, p.input_dtype(KernelInput::B));
    if (!(scale > 0) || scale > 1.7976931348623157e308 || zero < quantized_minimum(dtype) || zero > quantized_maximum(dtype)) return 0;
    const int64_t difference = quantized_code(input, uint64_t(row) * columns + column, dtype) - zero;
    if (p.output_dtype() == int(DType::Float64) || p.input_dtype(KernelInput::B) == int(DType::Float64))
        return double(difference) * scale;
    return double(float(difference) * float(scale));
}
template<class Reader>
ELMAN_AFFINE_HD void affine_quantize_unit(int unit, AffineQuantization operation, uint32_t* output, const uint32_t* input,
                                          const uint32_t* scales, const uint32_t* zeros,
                                          const KernelParams& p, Reader read) {
    const auto dtype = DType(p.output_dtype());
    const unsigned bits = quantized_storage_bits(dtype);
    if (bits < 8) {
        unsigned byte = 0;
        const unsigned per_byte = 8 / bits;
        for (unsigned part = 0; part < per_byte; ++part) {
            const int64_t index = int64_t(unit) * per_byte + part;
            if (index < p.output_count())
                byte |= unsigned(affine_encode_element(int(index), operation, input, scales, zeros, p, read)) << (part * bits);
        }
        reinterpret_cast<uint8_t*>(output)[unit] = uint8_t(byte);
    } else {
        const auto code = affine_encode_element(unit, operation, input, scales, zeros, p, read);
        if (dtype == DType::QInt32) reinterpret_cast<int32_t*>(output)[unit] = int32_t(code);
        else reinterpret_cast<uint8_t*>(output)[unit] = uint8_t(code);
    }
}
} // namespace elman
#undef ELMAN_AFFINE_HD
