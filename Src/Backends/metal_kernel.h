#ifndef ELMAN_METAL_KERNEL_H
#define ELMAN_METAL_KERNEL_H
#include <metal_stdlib>
using namespace metal;

inline float elman_metal_decode(uint word, int dtype) {
    return dtype == 0 ? as_type<float>(word) : (dtype == 1 ? float(as_type<int>(word)) : (word != 0u ? 1.0f : 0.0f));
}
inline int elman_metal_decode_int(uint word, int dtype) {
    return dtype == 0 ? int(as_type<float>(word)) : (dtype == 1 ? as_type<int>(word) : (word != 0u ? 1 : 0));
}
inline uint elman_metal_encode(float value, int dtype) {
    if (dtype == 0) return as_type<uint>(value);
    if (dtype == 2) return value != 0.0f ? 1u : 0u;
    if (isnan(value)) return 0u;
    if (value >= 2147483648.0f) return 2147483647u;
    if (value <= -2147483648.0f) return 2147483648u;
    return as_type<uint>(int(value));
}
inline uint elman_metal_encode_int(int value, int dtype) {
    return dtype == 0 ? as_type<uint>(float(value)) : (dtype == 1 ? as_type<uint>(value) : (value != 0 ? 1u : 0u));
}
inline float elman_real_pow(float base, float exponent) {
    // Preserve real-domain power semantics before using the positive-base intrinsic.
    if (exponent == 0.0f || base == 1.0f) return 1.0f;
    if (isnan(base) || isnan(exponent)) return as_type<float>(uint(0x7fc00000));
    float magnitude = abs(base);
    if (isinf(exponent)) {
        if (magnitude == 1.0f) return 1.0f;
        return (magnitude > 1.0f) == (exponent > 0.0f) ? as_type<float>(uint(0x7f800000)) : 0.0f;
    }
    bool integral = floor(exponent) == exponent;
    bool odd = integral && fmod(abs(exponent), 2.0f) == 1.0f;
    bool negative = (as_type<uint>(base) & 0x80000000u) != 0u;
    uint sign_bit = negative && odd ? 0x80000000u : 0u;
    if (magnitude == 0.0f) return as_type<float>(sign_bit | (exponent < 0.0f ? 0x7f800000u : 0u));
    if (isinf(magnitude)) return as_type<float>(sign_bit | (exponent > 0.0f ? 0x7f800000u : 0u));
    if (negative && !integral) return as_type<float>(uint(0x7fc00000));
    float result = pow(magnitude, exponent);
    return negative && odd ? -result : result;
}
#define P(index) (p[(index)])
#define F(index) as_type<float>(P(index))
#define A(index) elman_metal_decode(a[(index)],P(252))
#define B(index) elman_metal_decode(b[(index)],P(253))
#define C(index) elman_metal_decode(c[(index)],P(254))
#define IA(index) elman_metal_decode_int(a[(index)],P(252))
#define IB(index) elman_metal_decode_int(b[(index)],P(253))
#define IC(index) elman_metal_decode_int(c[(index)],P(254))
#define W(index,value) (out[(index)] = elman_metal_encode((value),P(255)))
#define WI(index,value) (out[(index)] = elman_metal_encode_int((value),P(255)))
#define COPY_A(destination,source) (out[(destination)] = P(252) == P(255) ? a[(source)] : elman_metal_encode(A(source),P(255)))
#define EL_INLINE inline
#define EL_CONTEXT_ARGS , constant int* p
#define EL_CONTEXT_PASS , p
#define EL_UINT uint
#define EL_SQRT(x) sqrt(x)
#define EL_EXP(x) exp(x)
#define EL_LOG(x) log(x)
#define EL_SIN(x) sin(x)
#define EL_COS(x) cos(x)
#define EL_POW(x,y) elman_real_pow((x),(y))
#define EL_TANH(x) tanh(x)
#define EL_FLOOR(x) floor(x)
#define EL_ABS(x) abs(x)
#define EL_MIN(x,y) (((x)<(y)) ? (x) : (y))
#define EL_MAX(x,y) (((x)>(y)) ? (x) : (y))
#define EL_INF as_type<float>(uint(0x7f800000))
#include "../Kernels/common.inc"
#endif
