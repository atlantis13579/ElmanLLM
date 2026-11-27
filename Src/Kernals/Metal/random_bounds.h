#pragma once

#include "Core/DType.h"
#include "Core/Float16.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace elman::metal {
// Clamp in the storage dtype, retaining a rounded lower endpoint when the
// interval collapses to one representable value.
inline float uniform_upper_bound(DType dtype, double low, double high) {
  float lower = float(low), upper = float(high);
  if (dtype == DType::Float16 || dtype == DType::BFloat16) {
    const bool half = dtype == DType::Float16;
    lower = half ? numeric::half_to_float(numeric::float_to_half(lower))
                 : numeric::bfloat_to_float(numeric::float_to_bfloat(lower));
    uint16_t bits = half ? numeric::float_to_half(upper) : numeric::float_to_bfloat(upper);
    upper = half ? numeric::half_to_float(bits) : numeric::bfloat_to_float(bits);
    // A rounded-down endpoint is already below high; stepping again can
    // exclude the only representable value in a narrow interval.
    if (double(upper) >= high) {
      if ((bits & 0x7fff) == 0) bits = 0x8001;
      else bits = uint16_t(bits + ((bits & 0x8000) ? 1 : -1));
      upper = half ? numeric::half_to_float(bits) : numeric::bfloat_to_float(bits);
    }
  } else if (double(upper) >= high) {
    upper = std::nextafter(upper, -std::numeric_limits<float>::infinity());
  }
  return std::max(lower, upper);
}
}  // namespace elman::metal
