#pragma once
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

#include "Core/Backend.h"
#include "Operators/attributes.h"
#include "Operators/tensor_axes.h"

namespace elman {
inline float cpu_float(uint32_t word) {
  float result;
  std::memcpy(&result, &word, 4);
  return result;
}
inline uint32_t cpu_bits(float value) {
  uint32_t result;
  std::memcpy(&result, &value, 4);
  return result;
}
inline float cpu_read(const uint32_t* data, int index, int dtype) {
  return dtype == 0 ? cpu_float(data[index]) : float(int32_t(data[index]));
}
inline int cpu_read_int(const uint32_t* data, int index, int dtype) {
  return dtype == 0 ? int(cpu_float(data[index])) : int32_t(data[index]);
}
inline void cpu_write(uint32_t* data, int index, float value, int dtype) {
  if (dtype == 0) {
    data[index] = cpu_bits(value);
  } else if (dtype == 2) {
    data[index] = value != 0.0f ? 1u : 0u;
  } else if (std::isnan(value)) {
    data[index] = 0;
  } else if (value >= 2147483648.0f) {
    data[index] = 2147483647u;
  } else if (value <= -2147483648.0f) {
    data[index] = 2147483648u;
  } else {
    data[index] = uint32_t(int32_t(value));
  }
}
inline void cpu_write_int(uint32_t* data, int index, int value, int dtype) {
  if (dtype == 0) {
    data[index] = cpu_bits(float(value));
  } else {
    data[index] = dtype == 2 ? (value != 0 ? 1u : 0u) : uint32_t(value);
  }
}
inline void cpu_copy_a(uint32_t* out, const uint32_t* a, int destination,
                       int source, const KernelParams& params) {
  if (params.input_dtype(KernelInput::A) == params.output_dtype()) {
    out[destination] = a[source];
  } else {
    cpu_write(out, destination,
              cpu_read(a, source, params.input_dtype(KernelInput::A)),
              params.output_dtype());
  }
}
inline float float_attribute(const KernelParams& params, int offset) {
  return cpu_float(uint32_t(params.attribute(offset)));
}
#define READ_A(index) cpu_read(a, index, p.input_dtype(KernelInput::A))
#define READ_B(index) cpu_read(b, index, p.input_dtype(KernelInput::B))
#define READ_C(index) cpu_read(c, index, p.input_dtype(KernelInput::C))
#define READ_INT_A(index) cpu_read_int(a, index, p.input_dtype(KernelInput::A))
#define READ_INT_B(index) cpu_read_int(b, index, p.input_dtype(KernelInput::B))
#define READ_INT_C(index) cpu_read_int(c, index, p.input_dtype(KernelInput::C))
#define WRITE_OUTPUT(index, value) \
  cpu_write(out, index, value, p.output_dtype())
#define WRITE_INT_OUTPUT(index, value) \
  cpu_write_int(out, index, value, p.output_dtype())
#define COPY_INPUT_A(index, source) cpu_copy_a(out, a, index, source, p)
#define EL_INLINE inline
#define EL_UINT uint32_t
#define EL_SQRT(x) std::sqrt(x)
#define EL_EXP(x) std::exp(x)
#define EL_LOG(x) std::log(x)
#define EL_SIN(x) std::sin(x)
#define EL_COS(x) std::cos(x)
#define EL_POW(x, y) std::pow(x, y)
#define EL_TANH(x) std::tanh(x)
#define EL_FLOOR(x) std::floor(x)
#define EL_ABS(x) std::fabs(x)
#define EL_MIN(x, y) ((x) < (y) ? (x) : (y))
#define EL_MAX(x, y) ((x) > (y) ? (x) : (y))
#define EL_INF std::numeric_limits<float>::infinity()
}  // namespace elman

#include "Operators/common.h"
