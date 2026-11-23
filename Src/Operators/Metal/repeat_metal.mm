#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_repeat(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Shape original = shape(p, 0);
  Shape repeated = shape(p);
  const int leading = int(repeated.count) - int(original.count);
  NSMutableArray<NSNumber*>* expanded = [NSMutableArray<NSNumber*> arrayWithCapacity:repeated.count];
  NSMutableArray<NSNumber*>* repetitions = [NSMutableArray<NSNumber*> arrayWithCapacity:repeated.count];
  for (int i = 0; i < int(repeated.count); ++i) {
    const int base = i < leading ? 1 : original[i - leading].intValue;
    if (!base) {
      return m.zeros(shape(p), a.dataType);
    }
    [expanded addObject:@(base)];
    [repetitions addObject:@(repeated[i].intValue / base)];
  }
  return [m.g tileTensor:m.reshape(a, expanded) withMultiplier:repetitions name:nil];
}
}  // namespace elman::metal
