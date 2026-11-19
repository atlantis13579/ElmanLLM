#pragma once
#include "Operators/integer_pointwise.h"

#ifdef __CUDACC__
#define ELMAN_EXTREMA_HD __host__ __device__
#else
#define ELMAN_EXTREMA_HD
#endif
namespace elman {
template<bool Maximum>
ELMAN_EXTREMA_HD inline uint64_t integer_extremum(uint64_t a, uint64_t b, int dtype) {
    // Sign extension followed by a sign-bit flip gives ordered unsigned keys.
    const uint64_t sign = integer_signed(dtype) ? uint64_t(1) << 63 : 0;
    const bool less = (a ^ sign) < (b ^ sign);
    return Maximum ? (less ? b : a) : (less ? a : b);
}
} // namespace elman
#undef ELMAN_EXTREMA_HD
