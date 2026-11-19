#include "Backends/MPSGraph.h"
#include "Backends/MetalDTypeSupport.h"

namespace elman::metal {
Tensor lower_complex(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  if (wide_words(DType(p.output_dtype()))) {
    auto pair_shape = [](Shape logical) { return [logical arrayByAddingObject:@2]; };
    auto target = pair_shape(shape(p));
    auto real = m.spread(m.reshape(a, pair_shape(shape(p, 0))), target);
    auto imag = m.spread(m.reshape(b, pair_shape(shape(p, 1))), target);
    return [m.g concatTensors:@[ real, imag ] dimension:int(target.count) - 1 name:nil];
  }
  return [m.g complexTensorWithRealTensor:m.float_work(a) imaginaryTensor:m.float_work(b) name:nil];
}
}  // namespace elman::metal
