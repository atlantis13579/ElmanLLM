#include "Backends/MPSGraph.h"

#include "Operators/float64_random_math.h"

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
  return m.select(m.equal(result, m.number(second, type)), m.number(first, type), result);
}
}  // namespace elman::metal
