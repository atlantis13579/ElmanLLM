#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"
#include "Operators/gelu_params.h"

namespace elman::metal {
Tensor lower_gelu(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  a = m.float_work(a);
  if (p.attribute(operator_attr::gelu::kExact)) {
    return m.mul(m.mul(a, m.number(.5)),
                 m.add(m.number(1), [m.g erfWithTensor:m.mul(a, m.number(.7071067811865475)) name:nil]));
  }
  auto cubic = m.mul(a, m.square(a));
  auto inner = m.mul(m.add(a, m.mul(cubic, m.number(.044715))), m.number(.7978845608028654));
  return m.mul(m.mul(a, m.number(.5)), m.add(m.number(1), [m.g tanhWithTensor:inner name:nil]));
}
}  // namespace elman::metal
