#include "Backends/MPSGraph.h"

#include "Core/Float16.h"
#include "Kernals/float64_random_math.h"

#include <cmath>
#include <limits>

namespace elman::metal {
MPSMatrixRandomDistributionDescriptor* uniform_distribution(const KernelParams& p) {
  const double first = float64_random_parameter(p, 0);
  const double second = float64_random_parameter(p, 1);
  if (float(first) == float(second)) {
    return nil;
  }
  return [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:float(first)
                                                                                 maximum:float(second)];
}

// Largest value strictly below high in the storage dtype. Clamping to this
// value keeps the half-open interval [low, high) without relying on equality
// selects, which MPSGraph mis-optimizes for BFloat16 upper endpoints.
static double next_below_high(DType dtype, double high) {
  if (dtype == DType::Float32) return std::nextafter(float(high), -std::numeric_limits<float>::infinity());
  const float value = float(high);
  if (dtype == DType::Float16) {
    uint16_t bits = numeric::float_to_half(value);
    if (bits == 0 || bits == 0x8000) bits = 0x8001;
    else bits += (bits & 0x8000) ? 1 : -1;
    return numeric::half_to_float(bits);
  }
  if (dtype == DType::BFloat16) {
    uint16_t bits = numeric::float_to_bfloat(value);
    if (bits == 0 || bits == 0x8000) bits = 0x8001;
    else bits += (bits & 0x8000) ? 1 : -1;
    return numeric::bfloat_to_float(bits);
  }
  return high;
}

Tensor lower_uniform(Graph m, const Operands&, const KernelParams& p) {
  Shape dims = shape(p);
  const DType dtype = DType(p.output_dtype());
  MPSDataType type = is_compute_float(dtype) ? storage_type(dtype) : MPSDataTypeFloat32;
  const double first = float64_random_parameter(p, 0);
  const double second = float64_random_parameter(p, 1);
  if (!m.random_samples) {
    return m.spread(m.number(first, type), dims);
  }
  Tensor result = m.cast(m.random_samples, type);
  return m.minimum(result, m.number(next_below_high(dtype, second), type));
}
}  // namespace elman::metal
