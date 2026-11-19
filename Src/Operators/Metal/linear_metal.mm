#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_linear(Graph m, const Operands& inputs, const KernelParams& p) {
  auto input = inputs[0];
  auto weight = inputs[1];
  auto bias = inputs[2];
  if (!input) {
    return bias ? m.spread(m.mul(bias, m.number(1, bias.dataType)), shape(p)) : m.filled(p, 0);
  }
  const int rank = weight.shape.count;
  weight = m.cast(m.transpose(weight, rank - 2, rank - 1), input.dataType);
  if (input.shape.count == 1) {
    input = m.reshape(input, @[ @1, input.shape[0] ]);
  }
  auto result = m.matmul(input, weight);
  return bias ? m.add(result, m.cast(bias, input.dataType)) : result;
}
}  // namespace elman::metal
