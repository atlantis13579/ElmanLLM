#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_flux_modulate(Graph m, const Operands& inputs, const KernelParams&) {
  auto input = inputs[0];
  auto shift = inputs[1];
  auto scale = inputs[2];
  auto x = m.float_work(input);
  auto centered = m.sub(x, m.mean(x, @[ @2 ]));
  auto inv = m.rsqrt(m.add(m.mean(m.square(centered), @[ @2 ]), m.number(1e-6)));
  auto norm = m.cast(m.mul(centered, inv), input.dataType);
  return m.add(m.mul(norm, m.add(scale, m.number(1, scale.dataType))), shift);
}
}  // namespace elman::metal
