#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_tril(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  const int rank = p.output_rank();
  auto target = shape(p);
  auto diagonal = m.sub(m.index(target, rank - 1), m.index(target, rank - 2));
  auto predicate = [m.g lessThanOrEqualToWithPrimaryTensor:diagonal
                                           secondaryTensor:m.integer(p.attribute(0))
                                                      name:nil];
  return m.select(predicate, a, m.number(0, a.dataType));
}
}  // namespace elman::metal
