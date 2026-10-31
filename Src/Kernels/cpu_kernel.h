#pragma once
#include "../Core/backend.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace elman {
inline float cpu_float(uint32_t word) { float result; std::memcpy(&result, &word, 4); return result; }
inline uint32_t cpu_bits(float value) { uint32_t result; std::memcpy(&result, &value, 4); return result; }
inline float cpu_read(const uint32_t* data, int index, int dtype) {
    return dtype == 0 ? cpu_float(data[index]) : float(int32_t(data[index]));
}
inline int cpu_read_int(const uint32_t* data, int index, int dtype) {
    return dtype == 0 ? int(cpu_float(data[index])) : int32_t(data[index]);
}
inline void cpu_write(uint32_t* data, int index, float value, int dtype) {
    if (dtype == 0) data[index] = cpu_bits(value);
    else if (dtype == 2) data[index] = value != 0.0f ? 1u : 0u;
    else if (std::isnan(value)) data[index] = 0;
    else if (value >= 2147483648.0f) data[index] = 2147483647u;
    else if (value <= -2147483648.0f) data[index] = 2147483648u;
    else data[index] = uint32_t(int32_t(value));
}
inline void cpu_write_int(uint32_t* data, int index, int value, int dtype) {
    if (dtype == 0) data[index] = cpu_bits(float(value));
    else data[index] = dtype == 2 ? (value != 0 ? 1u : 0u) : uint32_t(value);
}
#define P(index) (p.v[index])
#define F(index) cpu_float(uint32_t(P(index)))
#define A(index) cpu_read(a,index,P(252))
#define B(index) cpu_read(b,index,P(253))
#define C(index) cpu_read(c,index,P(254))
#define IA(index) cpu_read_int(a,index,P(252))
#define IB(index) cpu_read_int(b,index,P(253))
#define IC(index) cpu_read_int(c,index,P(254))
#define W(index,value) cpu_write(out,index,value,P(255))
#define WI(index,value) cpu_write_int(out,index,value,P(255))
#define COPY_A(index,source) do { if(P(252)==P(255)) out[index]=a[source]; else W(index,A(source)); } while(0)
#define EL_INLINE inline
#define EL_CONTEXT_ARGS , const KernelParams& p
#define EL_CONTEXT_PASS , p
#define EL_UINT uint32_t
#define EL_SQRT(x) std::sqrt(x)
#define EL_EXP(x) std::exp(x)
#define EL_LOG(x) std::log(x)
#define EL_SIN(x) std::sin(x)
#define EL_COS(x) std::cos(x)
#define EL_POW(x,y) std::pow(x,y)
#define EL_TANH(x) std::tanh(x)
#define EL_FLOOR(x) std::floor(x)
#define EL_ABS(x) std::fabs(x)
#define EL_MIN(x,y) ((x)<(y)?(x):(y))
#define EL_MAX(x,y) ((x)>(y)?(x):(y))
#define EL_INF std::numeric_limits<float>::infinity()
#include "common.inc"
}
