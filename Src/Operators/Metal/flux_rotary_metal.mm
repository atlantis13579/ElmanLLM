#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_flux_rotary(Graph m, const Operands& inputs, const KernelParams& p) {
  auto heads = inputs[0];
  auto table = inputs[1];
  const int batch = p.output_shape(0), count = p.output_shape(1), tokens = p.output_shape(2);
  auto x = m.reshape(m.float_work(heads), @[ @(batch), @(count), @(tokens), @64, @2 ]);
  auto t = m.reshape(table, @[ @(p.input_shape(KernelInput::B, 0)), @1, @(tokens), @64, @2 ]);
  auto even = m.slice(x, 4, 0, 1), odd = m.slice(x, 4, 1, 1);
  auto cs = m.slice(t, 4, 0, 1), sn = m.slice(t, 4, 1, 1);
  auto real = m.sub(m.mul(even, cs), m.mul(odd, sn));
  auto imag = m.add(m.mul(odd, cs), m.mul(even, sn));
  return m.cast(m.reshape([m.g concatTensors:@[ real, imag ] dimension:4 name:nil], heads.shape),
                heads.dataType);
}
}  // namespace elman::metal
