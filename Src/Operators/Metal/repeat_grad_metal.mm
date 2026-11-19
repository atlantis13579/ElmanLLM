#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_repeat_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  if (!a) {
    return m.filled(p, 0);
  }
  auto original = shape(p);
  auto repeated = shape(p, 0);
  const int leading = int(repeated.count) - int(original.count);
  auto expanded = [NSMutableArray<NSNumber*> arrayWithCapacity:repeated.count];
  auto repetitions = [NSMutableArray<NSNumber*> arrayWithCapacity:repeated.count];
  for (int i = 0; i < int(repeated.count); ++i) {
    const int base = i < leading ? 1 : original[i - leading].intValue;
    if (!base) {
      return m.zeros(shape(p), a.dataType);
    }
    [expanded addObject:@(base)];
    [repetitions addObject:@(repeated[i].intValue / base)];
  }
  return [m.g tileGradientWithIncomingGradientTensor:a
                                        sourceTensor:m.zeros(expanded, a.dataType)
                                      withMultiplier:repetitions
                                                name:nil];
}
}  // namespace elman::metal
