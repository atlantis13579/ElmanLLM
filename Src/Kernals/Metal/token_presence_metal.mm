#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_token_presence(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const int rows = p.output_shape(0), width = p.output_shape(1);
  const int count = rows * p.input_shape(KernelInput::A, 1);
  if (!count) {
    return m.zeros(shape(p), MPSDataTypeInt32);
  }
  Tensor ids = m.cast(a, MPSDataTypeInt32);
  Tensor words = m.select(m.bounded(ids, width - 1), ids, m.integer(width - 1));
  Tensor offsets = m.add(m.mul(m.index(a.shape, 0), m.integer(width)), words);
  NSArray<NSNumber*>* flat = @[ @(count) ];
  return [m.g scatterWithUpdatesTensor:m.spread(m.integer(1), flat)
                         indicesTensor:m.reshape(offsets, flat)
                                 shape:@[ @(p.output_count()) ]
                                  axis:0
                                  mode:MPSGraphScatterModeMax
                                  name:nil];
}
}  // namespace elman::metal
