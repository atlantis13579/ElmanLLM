#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_permute(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  const int rank = p.output_rank();
  auto* order = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  for (int i = 0; i < rank; ++i) {
    [order addObject:@(p.attribute(i))];
  }
  return [m.g transposeTensor:a permutation:order name:nil];
}
}  // namespace elman::metal
