#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_triu(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const int rank = p.output_rank();
  Shape target = shape(p);
  Tensor diagonal = m.sub(m.index(target, rank - 1), m.index(target, rank - 2));
  Tensor predicate = [m.g greaterThanOrEqualToWithPrimaryTensor:diagonal
                                              secondaryTensor:m.integer(p.attribute(0))
                                                         name:nil];
  return m.select(predicate, a, m.number(0, a.dataType));
}
}  // namespace elman::metal
