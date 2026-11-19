#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cstdint>

#include "Core/Backend.h"
#include "Backends/CudaStream.h"
#include "Operators/attributes.h"
#include "Operators/tensor_axes.h"

__device__ inline uint32_t elman_cuda_load(const uint32_t* data, int index, int dtype) {
  if (dtype == 3 || dtype == 4) return reinterpret_cast<const uint16_t*>(data)[index];
  return dtype == 2 ? reinterpret_cast<const uint8_t*>(data)[index] : data[index];
}
__device__ inline void elman_cuda_store(uint32_t* data, int index, uint32_t word, int dtype) {
  if (dtype == 2) reinterpret_cast<uint8_t*>(data)[index] = uint8_t(word);
  else if (dtype == 3 || dtype == 4) reinterpret_cast<uint16_t*>(data)[index] = uint16_t(word);
  else data[index] = word;
}
__device__ inline void elman_cuda_copy_element(uint32_t* out, const uint32_t* source, int destination, int index, int dtype) {
  elman_cuda_store(out, destination, elman_cuda_load(source, index, dtype), dtype);
}
__device__ inline float elman_cuda_decode(std::uint32_t word, int dtype) {
  if (dtype == 3) return __half2float(__ushort_as_half(uint16_t(word)));
  if (dtype == 4) return __bfloat162float(__ushort_as_bfloat16(uint16_t(word)));
  return dtype == 0 ? __uint_as_float(word)
                    : (dtype == 1 ? float(static_cast<int>(word))
                                  : (word != 0 ? 1.0f : 0.0f));
}
__device__ inline int elman_cuda_decode_int(std::uint32_t word, int dtype) {
  if (dtype == 3 || dtype == 4) return int(elman_cuda_decode(word, dtype));
  return dtype == 0
             ? int(__uint_as_float(word))
             : (dtype == 1 ? static_cast<int>(word) : (word != 0 ? 1 : 0));
}
__device__ inline std::uint32_t elman_cuda_encode(float value, int dtype) {
  if (dtype == 3) return __half_as_ushort(__float2half_rn(value));
  if (dtype == 4) return __bfloat16_as_ushort(__float2bfloat16_rn(value));
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
  if (dtype == 3 || dtype == 4) return elman_cuda_encode(float(value), dtype);
  return dtype == 0 ? __float_as_uint(float(value))
                    : (dtype == 1 ? static_cast<std::uint32_t>(value)
                                  : (value != 0 ? 1u : 0u));
}
__device__ inline void elman_cuda_copy_a(std::uint32_t* out,
                                         const std::uint32_t* a,
                                         int destination, int source,
                                         const elman::KernelParams& params) {
  uint32_t value =
      params.input_dtype(elman::KernelInput::A) == params.output_dtype()
          ? elman_cuda_load(a, source, params.input_dtype(elman::KernelInput::A))
          : elman_cuda_encode(
                elman_cuda_decode(elman_cuda_load(a, source, params.input_dtype(elman::KernelInput::A)),
                                  params.input_dtype(elman::KernelInput::A)),
                params.output_dtype());
  elman_cuda_store(out, destination, value, params.output_dtype());
}

namespace elman {
__device__ inline float float_attribute(const KernelParams& params,
                                        int offset) {
  return __int_as_float(params.attribute(offset));
}
}  // namespace elman

#define READ_A(index) \
  elman_cuda_decode(elman_cuda_load(a, (index), p.input_dtype(elman::KernelInput::A)), p.input_dtype(elman::KernelInput::A))
#define READ_B(index) \
  elman_cuda_decode(elman_cuda_load(b, (index), p.input_dtype(elman::KernelInput::B)), p.input_dtype(elman::KernelInput::B))
#define READ_C(index) \
  elman_cuda_decode(elman_cuda_load(c, (index), p.input_dtype(elman::KernelInput::C)), p.input_dtype(elman::KernelInput::C))
#define READ_INT_A(index) \
  elman_cuda_decode_int(elman_cuda_load(a, (index), p.input_dtype(elman::KernelInput::A)), p.input_dtype(elman::KernelInput::A))
#define READ_INT_B(index) \
  elman_cuda_decode_int(elman_cuda_load(b, (index), p.input_dtype(elman::KernelInput::B)), p.input_dtype(elman::KernelInput::B))
#define READ_INT_C(index) \
  elman_cuda_decode_int(elman_cuda_load(c, (index), p.input_dtype(elman::KernelInput::C)), p.input_dtype(elman::KernelInput::C))
#define WRITE_OUTPUT(index, value) \
  elman_cuda_store(out, (index), elman_cuda_encode((value), p.output_dtype()), p.output_dtype())
#define WRITE_INT_OUTPUT(index, value) \
  elman_cuda_store(out, (index), elman_cuda_encode_int((value), p.output_dtype()), p.output_dtype())
#define COPY_INPUT_A(destination, source) \
  elman_cuda_copy_a(out, a, destination, source, p)
#define COPY_TYPED_ELEMENT(out, source, destination, index, dtype) elman_cuda_copy_element(out, source, destination, index, dtype)
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
#define EL_MIN(x, y) (isnan(x) || (x) < (y) ? (x) : (y))
#define EL_MAX(x, y) (isnan(x) || (x) > (y) ? (x) : (y))
#define EL_INF __int_as_float(0x7f800000)
#include "Operators/common.h"
