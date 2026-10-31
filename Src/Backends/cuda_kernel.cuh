#pragma once
#include "Core/backend.h"
#include <cuda_runtime.h>
#include <cstdint>

__device__ inline float elman_cuda_decode(std::uint32_t word, int dtype) {
    return dtype == 0 ? __uint_as_float(word) : (dtype == 1 ? float(static_cast<int>(word)) : (word != 0 ? 1.0f : 0.0f));
}
__device__ inline int elman_cuda_decode_int(std::uint32_t word, int dtype) {
    return dtype == 0 ? int(__uint_as_float(word)) : (dtype == 1 ? static_cast<int>(word) : (word != 0 ? 1 : 0));
}
__device__ inline std::uint32_t elman_cuda_encode(float value, int dtype) {
    if (dtype == 0) return __float_as_uint(value);
    if (dtype == 2) return value != 0.0f ? 1u : 0u;
    if (isnan(value)) return 0u;
    if (value >= 2147483648.0f) return 2147483647u;
    if (value <= -2147483648.0f) return 2147483648u;
    return static_cast<std::uint32_t>(int(value));
}
__device__ inline std::uint32_t elman_cuda_encode_int(int value, int dtype) {
    return dtype == 0 ? __float_as_uint(float(value)) : (dtype == 1 ? static_cast<std::uint32_t>(value) : (value != 0 ? 1u : 0u));
}

#define P(index) (p.v[(index)])
#define F(index) __int_as_float(P(index))
#define A(index) elman_cuda_decode(a[(index)], P(252))
#define B(index) elman_cuda_decode(b[(index)], P(253))
#define C(index) elman_cuda_decode(c[(index)], P(254))
#define IA(index) elman_cuda_decode_int(a[(index)], P(252))
#define IB(index) elman_cuda_decode_int(b[(index)], P(253))
#define IC(index) elman_cuda_decode_int(c[(index)], P(254))
#define W(index,value) (out[(index)] = elman_cuda_encode((value), P(255)))
#define WI(index,value) (out[(index)] = elman_cuda_encode_int((value), P(255)))
#define COPY_A(destination,source) (out[(destination)] = P(252) == P(255) ? a[(source)] : elman_cuda_encode(A(source), P(255)))
#define EL_INLINE __device__ inline
#define EL_CONTEXT_ARGS , const elman::KernelParams& p
#define EL_CONTEXT_PASS , p
#define EL_UINT std::uint32_t
#define EL_SQRT(x) sqrtf(x)
#define EL_EXP(x) expf(x)
#define EL_LOG(x) logf(x)
#define EL_SIN(x) sinf(x)
#define EL_COS(x) cosf(x)
#define EL_POW(x,y) powf((x),(y))
#define EL_TANH(x) tanhf(x)
#define EL_FLOOR(x) floorf(x)
#define EL_ABS(x) fabsf(x)
#define EL_MIN(x,y) (((x)<(y)) ? (x) : (y))
#define EL_MAX(x,y) (((x)>(y)) ? (x) : (y))
#define EL_INF __int_as_float(0x7f800000)
#include "Kernels/common.inc"
