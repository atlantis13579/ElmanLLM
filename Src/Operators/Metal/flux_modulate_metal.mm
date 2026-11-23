#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_flux_modulate(Graph m, const Operands& inputs, const KernelParams&) {
  Tensor input = inputs[0];
  Tensor shift = inputs[1];
  Tensor scale = inputs[2];
  Tensor x = m.float_work(input);
  Tensor centered = m.sub(x, m.mean(x, @[ @2 ]));
  Tensor inv = m.rsqrt(m.add(m.mean(m.square(centered), @[ @2 ]), m.number(1e-6)));
  Tensor norm = m.cast(m.mul(centered, inv), input.dataType);
  return m.add(m.mul(norm, m.add(scale, m.number(1, scale.dataType))), shift);
}
}  // namespace elman::metal
