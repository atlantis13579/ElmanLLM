#pragma once
#include "Core/Backend.h"
#include "Core/Float16.h"
#include "Operators/attributes.h"

#ifdef __CUDACC__
#define ELMAN_DROPOUT_HD __host__ __device__
#else
#define ELMAN_DROPOUT_HD
#endif
namespace elman {
ELMAN_DROPOUT_HD inline float floating_dropout_scale(DType dtype, double probability, bool backward) {
    if (probability == 1.) return 0.f;
#if defined(__CUDA_ARCH__)
    // CUDA backward receives the original Double reciprocal from autograd.
    // Forward first narrows the keep probability to its accumulation dtype.
    if (backward) return float(1. / (1. - probability));
#endif
    float scale = 1.f / float(1. - probability);
#if !defined(__CUDA_ARCH__)
    // CPU dropout saves a noise tensor in the input dtype for both passes.
    if (dtype == DType::Float16) scale = numeric::half_to_float(numeric::float_to_half(scale));
    else if (dtype == DType::BFloat16) scale = numeric::bfloat_to_float(numeric::float_to_bfloat(scale));
#endif
    return scale;
}
} // namespace elman
#undef ELMAN_DROPOUT_HD
