#pragma once
#include <cuda_runtime.h>

#include <cstdint>

#include "Core/Backend.h"
#include "Operators/attributes.h"
#include "Operators/tensor_axes.h"

__device__ inline float elman_cuda_decode(std::uint32_t word, int dtype) {
  return dtype == 0 ? __uint_as_float(word)
                    : (dtype == 1 ? float(static_cast<int>(word))
                                  : (word != 0 ? 1.0f : 0.0f));
}
__device__ inline int elman_cuda_decode_int(std::uint32_t word, int dtype) {
  return dtype == 0
             ? int(__uint_as_float(word))
             : (dtype == 1 ? static_cast<int>(word) : (word != 0 ? 1 : 0));
}
__device__ inline std::uint32_t elman_cuda_encode(float value, int dtype) {
  if (dtype == 0) {
    return __float_as_uint(value);
  }
  if (dtype == 2) {
    return value != 0.0f ? 1u : 0u;
  }
  if (isnan(value)) {
    return 0u;
  }
  if (value >= 2147483648.0f) {
    return 2147483647u;
  }
  if (value <= -2147483648.0f) {
    return 2147483648u;
  }
  return static_cast<std::uint32_t>(int(value));
}
__device__ inline std::uint32_t elman_cuda_encode_int(int value, int dtype) {
  return dtype == 0 ? __float_as_uint(float(value))
                    : (dtype == 1 ? static_cast<std::uint32_t>(value)
                                  : (value != 0 ? 1u : 0u));
}
__device__ inline void elman_cuda_copy_a(std::uint32_t* out,
                                         const std::uint32_t* a,
                                         int destination, int source,
                                         const elman::KernelParams& params) {
  out[destination] =
      params.input_dtype(elman::KernelInput::A) == params.output_dtype()
          ? a[source]
          : elman_cuda_encode(
                elman_cuda_decode(a[source],
                                  params.input_dtype(elman::KernelInput::A)),
                params.output_dtype());
}

namespace elman {
__device__ inline float float_attribute(const KernelParams& params,
                                        int offset) {
  return __int_as_float(params.attribute(offset));
}
}  // namespace elman

#define READ_A(index) \
  elman_cuda_decode(a[(index)], p.input_dtype(elman::KernelInput::A))
#define READ_B(index) \
  elman_cuda_decode(b[(index)], p.input_dtype(elman::KernelInput::B))
#define READ_C(index) \
  elman_cuda_decode(c[(index)], p.input_dtype(elman::KernelInput::C))
#define READ_INT_A(index) \
  elman_cuda_decode_int(a[(index)], p.input_dtype(elman::KernelInput::A))
#define READ_INT_B(index) \
  elman_cuda_decode_int(b[(index)], p.input_dtype(elman::KernelInput::B))
#define READ_INT_C(index) \
  elman_cuda_decode_int(c[(index)], p.input_dtype(elman::KernelInput::C))
#define WRITE_OUTPUT(index, value) \
  (out[(index)] = elman_cuda_encode((value), p.output_dtype()))
#define WRITE_INT_OUTPUT(index, value) \
  (out[(index)] = elman_cuda_encode_int((value), p.output_dtype()))
#define COPY_INPUT_A(destination, source) \
  elman_cuda_copy_a(out, a, destination, source, p)
#define EL_INLINE __device__ inline
#define EL_UINT std::uint32_t
#define EL_SQRT(x) sqrtf(x)
#define EL_EXP(x) expf(x)
#define EL_LOG(x) logf(x)
#define EL_SIN(x) sinf(x)
#define EL_COS(x) cosf(x)
#define EL_POW(x, y) powf((x), (y))
#define EL_TANH(x) tanhf(x)
#define EL_FLOOR(x) floorf(x)
#define EL_ABS(x) fabsf(x)
#define EL_MIN(x, y) (((x) < (y)) ? (x) : (y))
#define EL_MAX(x, y) (((x) > (y)) ? (x) : (y))
#define EL_INF __int_as_float(0x7f800000)
#include "Operators/common.h"
