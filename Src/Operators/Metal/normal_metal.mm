#include "Backends/MPSGraph.h"

#include "Operators/float64_random_math.h"

namespace elman::metal {
MPSMatrixRandomDistributionDescriptor* normal_distribution(const KernelParams& p) {
  const double first = float64_random_parameter(p, 0);
  const double second = float64_random_parameter(p, 1);
  if (float(second) == 0) {
    return nil;
  }
  return [MPSMatrixRandomDistributionDescriptor normalDistributionDescriptorWithMean:float(first)
                                                                   standardDeviation:float(second)];
}

Tensor lower_normal(Graph m, const Operands&, const KernelParams& p) {
  Shape dims = shape(p);
  const DType dtype = DType(p.output_dtype());
  MPSDataType type = is_compute_float(dtype) ? storage_type(dtype) : MPSDataTypeFloat32;
  const double first = float64_random_parameter(p, 0);
  if (!m.random_samples) {
    return m.spread(m.number(first, type), dims);
  }
  return m.cast(m.random_samples, type);
}
}  // namespace elman::metal
