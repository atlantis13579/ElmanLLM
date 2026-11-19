#pragma once
#include "Core/Backend.h"

namespace elman {
enum class CudaConvolution { InputGradient, WeightGradient, Forward };

bool try_pointwise_conv_gradient_cuda(CudaConvolution operation, uint32_t* out,
                                       const uint32_t* a, const uint32_t* b, const KernelParams& p);
#ifdef ELMAN_HAS_CUDNN
bool try_convolution_cuda(CudaConvolution operation, uint32_t* out, const uint32_t* a, const uint32_t* b,
                          const KernelParams& p);
#else
inline bool try_convolution_cuda(CudaConvolution, uint32_t*, const uint32_t*, const uint32_t*,
                                 const KernelParams&) { return false; }
#endif
}  // namespace elman
