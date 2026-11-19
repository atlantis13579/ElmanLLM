#pragma once

#include <cstdint>
#include <cstring>

namespace elman {
namespace numeric {

inline uint32_t float_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}
inline float bits_float(uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}
inline uint32_t rounded_shift(uint32_t value, unsigned shift) {
    const uint32_t base = value >> shift;
    const uint32_t remainder = value & ((uint32_t(1) << shift) - 1);
    const uint32_t midpoint = uint32_t(1) << (shift - 1);
    return base + (remainder > midpoint || (remainder == midpoint && (base & 1)));
}
inline uint16_t float_to_half(float value) {
    const uint32_t bits = float_bits(value);
    const uint16_t sign = uint16_t((bits >> 16) & 0x8000);
    const uint32_t fraction = bits & 0x7fffff;
    const int exponent = int((bits >> 23) & 255);
    if (exponent == 255) return uint16_t(sign | (fraction ? 0x7e00 : 0x7c00));
    const int adjusted = exponent - 127 + 15;
    if (adjusted >= 31) return uint16_t(sign | 0x7c00);
    if (adjusted <= 0) {
        if (adjusted < -10) return sign;
        return uint16_t(sign | rounded_shift(fraction | 0x800000, unsigned(14 - adjusted)));
    }
    return uint16_t(sign | ((uint32_t(adjusted) << 10) + rounded_shift(fraction, 13)));
}
inline float half_to_float(uint16_t value) {
    const uint32_t sign = uint32_t(value & 0x8000) << 16;
    uint32_t fraction = value & 0x3ff;
    const int exponent = (value >> 10) & 31;
    if (exponent == 31) return bits_float(sign | 0x7f800000 | (fraction << 13));
    if (exponent) return bits_float(sign | (uint32_t(exponent + 112) << 23) | (fraction << 13));
    if (!fraction) return bits_float(sign);
    int adjusted = -14;
    while (!(fraction & 0x400)) { fraction <<= 1; --adjusted; }
    return bits_float(sign | (uint32_t(adjusted + 127) << 23) | ((fraction & 0x3ff) << 13));
}
inline uint16_t float_to_bfloat(float value) {
    const uint32_t bits = float_bits(value);
    // Map both NaN signs to the same BF16 encoding.
    if ((bits & 0x7fffffff) > 0x7f800000) return 0xffff;
    return uint16_t((bits + 0x7fff + ((bits >> 16) & 1)) >> 16);
}
inline float bfloat_to_float(uint16_t value) { return bits_float(uint32_t(value) << 16); }

} // namespace numeric
} // namespace elman
