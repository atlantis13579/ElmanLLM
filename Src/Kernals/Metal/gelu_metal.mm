#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"
#include "Kernals/gelu_params.h"

namespace elman::metal {
Tensor lower_gelu(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  a = m.float_work(a);
  if (p.attribute(kernal_attr::gelu::kExact)) {
    return m.mul(m.mul(a, m.number(.5)),
                 m.add(m.number(1), [m.g erfWithTensor:m.mul(a, m.number(.7071067811865475)) name:nil]));
  }
  Tensor cubic = m.mul(a, m.square(a));
  Tensor inner = m.mul(m.add(a, m.mul(cubic, m.number(.044715))), m.number(.7978845608028654));
  return m.mul(m.mul(a, m.number(.5)), m.add(m.number(1), [m.g tanhWithTensor:inner name:nil]));
}
}  // namespace elman::metal
