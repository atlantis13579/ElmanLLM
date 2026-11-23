#pragma once

#include "Core/DType.h"
#include <cstdint>

#ifdef __CUDACC__
#define ELMAN_LOW_FLOAT_HD __host__ __device__
#else
#define ELMAN_LOW_FLOAT_HD
#endif

namespace elman {
inline bool is_differentiable_float(DType dtype) {
    return is_arithmetic_float(dtype) || is_low_precision_float(dtype);
}
namespace numeric {

ELMAN_LOW_FLOAT_HD inline uint32_t low_float_bits(float value) {
    union { float value; uint32_t bits; } result;
    result.value = value;
    return result.bits;
}
ELMAN_LOW_FLOAT_HD inline float low_bits_float(uint32_t bits) {
    union { float value; uint32_t bits; } result;
    result.bits = bits;
    return result.value;
}
ELMAN_LOW_FLOAT_HD inline uint32_t low_float_round(uint32_t value, unsigned shift) {
    if (shift >= 32) return 0;
    const uint32_t base = value >> shift;
    const uint32_t rest = value & ((uint32_t(1) << shift) - 1);
    const uint32_t half = uint32_t(1) << (shift - 1);
    return base + (rest > half || (rest == half && (base & 1)));
}
ELMAN_LOW_FLOAT_HD inline uint8_t float_to_float8(float value, DType dtype) {
    const uint32_t bits = low_float_bits(value);
    const uint32_t fraction = bits & 0x7fffff;
    const int exponent = int((bits >> 23) & 255);
    if (dtype == DType::Float8E8M0FNU) {
        // E8M0 rounds the magnitude to a power of two and uses exponent zero for zero input.
        if (exponent == 255) return 255;
        return uint8_t(exponent + (fraction > 0x400000 || (fraction == 0x400000 && exponent != 0)));
    }
    const bool e4 = dtype == DType::Float8E4M3FN || dtype == DType::Float8E4M3FNUZ;
    const bool unsigned_zero = dtype == DType::Float8E4M3FNUZ || dtype == DType::Float8E5M2FNUZ;
    const int mantissa = e4 ? 3 : 2;
    const int bias = (e4 ? 7 : 15) + int(unsigned_zero);
    const uint8_t sign = uint8_t((bits >> 24) & 128);
    if (exponent == 255) {
        if (unsigned_zero) return 128;
        return uint8_t(sign | (e4 || fraction ? 127 : 124));
    }
    const int adjusted = exponent - 127 + bias;
    uint32_t code;
    if (adjusted <= 0) {
        code = low_float_round(fraction | 0x800000, unsigned(24 - mantissa - adjusted));
    } else {
        code = (uint32_t(adjusted) << mantissa) + low_float_round(fraction, unsigned(23 - mantissa));
    }
    if (unsigned_zero) {
        if (code > 127) return 128;
        if (!code) return 0;
    } else if (e4) {
        if (code > 126) code = 127;
    } else if (code > 124) code = 124;
    return uint8_t(sign | code);
}
ELMAN_LOW_FLOAT_HD inline float float8_to_float(uint8_t value, DType dtype) {
    if (dtype == DType::Float8E8M0FNU) {
        return low_bits_float(value == 255 ? 0x7fc00000 : value == 0 ? 0x00400000 : uint32_t(value) << 23);
    }
    const bool e4 = dtype == DType::Float8E4M3FN || dtype == DType::Float8E4M3FNUZ;
    const bool unsigned_zero = dtype == DType::Float8E4M3FNUZ || dtype == DType::Float8E5M2FNUZ;
    if (unsigned_zero && value == 128) return low_bits_float(0x7fc00000);
    const int mantissa = e4 ? 3 : 2;
    const int bias = (e4 ? 7 : 15) + int(unsigned_zero);
    const uint32_t sign = uint32_t(value & 128) << 24;
    int exponent = (value & 127) >> mantissa;
    uint32_t fraction = value & ((1u << mantissa) - 1);
    if (!unsigned_zero && ((e4 && (value & 127) == 127) || (!e4 && exponent == 31)))
        return low_bits_float(sign | (fraction ? 0x7fc00000 : 0x7f800000));
    if (exponent) return low_bits_float(sign | (uint32_t(exponent - bias + 127) << 23) | (fraction << (23 - mantissa)));
    if (!fraction) return low_bits_float(sign);
    exponent = 1 - bias;
    while (!(fraction & (1u << mantissa))) { fraction <<= 1; --exponent; }
    return low_bits_float(sign | (uint32_t(exponent + 127) << 23) | ((fraction & ((1u << mantissa) - 1)) << (23 - mantissa)));
}
template<class Float>
ELMAN_LOW_FLOAT_HD inline uint8_t encode_float4(Float value, uint8_t sign) {
    if (value != value) return sign;
    const Float magnitude = sign ? -value : value;
    // E2M1 midpoint ties select the even low encoding bit.
    const uint8_t code = magnitude <= .25f ? 0 : magnitude < .75f ? 1 :
        magnitude <= 1.25f ? 2 : magnitude < 1.75f ? 3 : magnitude <= 2.5f ? 4 :
        magnitude < 3.5f ? 5 : magnitude <= 5.f ? 6 : 7;
    return uint8_t(sign | code);
}
ELMAN_LOW_FLOAT_HD inline uint8_t float_to_float4(float value) {
    return encode_float4(value, uint8_t((low_float_bits(value) >> 28) & 8));
}
ELMAN_LOW_FLOAT_HD inline uint8_t double_to_float4(double value) {
    union { double value; uint64_t bits; } encoded;
    encoded.value = value;
    return encode_float4(value, uint8_t((encoded.bits >> 60) & 8));
}
ELMAN_LOW_FLOAT_HD inline float float4_to_float(uint8_t code) {
    const unsigned exponent = (code >> 1) & 3;
    const float fraction = float(code & 1) * .5f;
    const float magnitude = exponent ? (1.f + fraction) * float(1u << (exponent - 1)) : fraction;
    return code & 8 ? -magnitude : magnitude;
}
ELMAN_LOW_FLOAT_HD inline double float4_to_double(uint8_t code) {
    return float4_to_float(code);
}

} // namespace numeric
} // namespace elman
#undef ELMAN_LOW_FLOAT_HD
