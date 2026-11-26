#pragma once
#include "Kernals/low_precision_loss.h"
#include "Kernals/low_precision_pointwise_math.h"
#ifdef __CUDACC__
#define ELMAN_LOW_LOSS_HD __device__
#else
#define ELMAN_LOW_LOSS_HD
#endif
namespace elman {
ELMAN_LOW_LOSS_HD inline float low_mse_derivative(int i, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p, int count) {
    const DType dtype = DType(p.input_dtype(KernelInput::A));
    const float difference = low_precision_value(a, i, dtype) - low_precision_value(b, i, dtype);
    const float upstream = low_precision_value(c, 0, DType(p.input_dtype(KernelInput::C)));
    const float factor = 2.f / float(count);
#ifdef __CUDACC__
    return __fmul_rn(__fmul_rn(factor, difference), upstream);
#else
    return (factor * difference) * upstream;
#endif
}
} // namespace elman
#undef ELMAN_LOW_LOSS_HD
