#ifndef ELMAN_COMPUTE_KERNEL_HLSLI
#define ELMAN_COMPUTE_KERNEL_HLSLI
cbuffer ElmanParameters : register(b0) { int4 elman_parameters[64]; };
StructuredBuffer<uint> input_a : register(t0);
StructuredBuffer<uint> input_b : register(t1);
StructuredBuffer<uint> input_c : register(t2);
RWStructuredBuffer<uint> output_values : register(u0);

int elman_param(int index) { return elman_parameters[index / 4][index % 4]; }
float elman_decode(uint word, int dtype) {
    return dtype == 0 ? asfloat(word) : (dtype == 1 ? float(asint(word)) : (word != 0u ? 1.0f : 0.0f));
}
int elman_decode_int(uint word, int dtype) {
    return dtype == 0 ? int(asfloat(word)) : (dtype == 1 ? asint(word) : (word != 0u ? 1 : 0));
}
uint elman_encode(float value, int dtype) {
    if (dtype == 0) return asuint(value);
    if (dtype == 2) return value != 0.0f ? 1u : 0u;
    if (isnan(value)) return 0u;
    if (value >= 2147483648.0f) return 2147483647u;
    if (value <= -2147483648.0f) return 2147483648u;
    return asuint(int(value));
}
uint elman_encode_int(int value, int dtype) {
    return dtype == 0 ? asuint(float(value)) : (dtype == 1 ? asuint(value) : (value != 0 ? 1u : 0u));
}
float elman_real_pow(float base, float exponent) {
    // HLSL pow rejects every negative base and leaves zero-to-zero GPU-dependent.
    if (exponent == 0.0f || base == 1.0f) return 1.0f;
    if (isnan(base) || isnan(exponent)) return asfloat(0x7fc00000u);
    float magnitude = abs(base);
    if (isinf(exponent)) {
        if (magnitude == 1.0f) return 1.0f;
        return (magnitude > 1.0f) == (exponent > 0.0f) ? asfloat(0x7f800000u) : 0.0f;
    }
    bool integral = floor(exponent) == exponent;
    bool odd = integral && fmod(abs(exponent), 2.0f) == 1.0f;
    bool negative = (asuint(base) & 0x80000000u) != 0u;
    uint sign_bit = negative && odd ? 0x80000000u : 0u;
    if (magnitude == 0.0f) return asfloat(sign_bit | (exponent < 0.0f ? 0x7f800000u : 0u));
    if (isinf(magnitude)) return asfloat(sign_bit | (exponent > 0.0f ? 0x7f800000u : 0u));
    if (negative && !integral) return asfloat(0x7fc00000u);
    float result = pow(magnitude, exponent);
    return negative && odd ? -result : result;
}
#define P(index) elman_param(index)
#define F(index) asfloat(P(index))
#define A(index) elman_decode(input_a[(index)],P(252))
#define B(index) elman_decode(input_b[(index)],P(253))
#define C(index) elman_decode(input_c[(index)],P(254))
#define IA(index) elman_decode_int(input_a[(index)],P(252))
#define IB(index) elman_decode_int(input_b[(index)],P(253))
#define IC(index) elman_decode_int(input_c[(index)],P(254))
#define W(index,value) (output_values[(index)] = elman_encode((value),P(255)))
#define WI(index,value) (output_values[(index)] = elman_encode_int((value),P(255)))
#define COPY_A(destination,source) (output_values[(destination)] = P(252) == P(255) ? input_a[(source)] : elman_encode(A(source),P(255)))
#define EL_INLINE
#define EL_CONTEXT_ARGS
#define EL_CONTEXT_PASS
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
#define EL_INF asfloat(0x7f800000)
#include "../Kernels/common.inc"
#endif
