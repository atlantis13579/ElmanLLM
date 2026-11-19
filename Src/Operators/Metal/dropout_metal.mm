#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"
#include "Operators/float64_random_math.h"

namespace elman::metal {
MPSMatrixRandomDistributionDescriptor* dropout_distribution(const KernelParams& p) {
  const double first = float64_random_parameter(p, 0);
  if (!p.attribute(operator_attr::dropout::kTraining) || first == 0 || first == 1) {
    return nil;
  }
  return [MPSMatrixRandomDistributionDescriptor uniformDistributionDescriptorWithMinimum:0 maximum:1];
}

Tensor lower_dropout(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  const double first = float64_random_parameter(p, 0);
  if (!p.attribute(operator_attr::dropout::kTraining) || first == 0) {
    return a;
  }
  if (first == 1) {
    return m.mul(a, m.number(0, a.dataType));
  }
  auto noise = m.select(m.less(m.random_samples, m.number(first)), m.number(0, a.dataType),
                        m.number(1 / (1 - first), a.dataType));
  return m.mul(a, noise);
}
}  // namespace elman::metal
