#include <algorithm>

#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_log_softmax(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  const int selected = std::max(p.attribute(operator_attr::reduction::kAxis), 0);
  auto work = m.float_work(a);
  auto shifted = m.sub(work, m.max(work, @[ @(selected) ]));
  return m.sub(shifted, m.log(m.sum(m.exp(shifted), @[ @(selected) ])));
}
}  // namespace elman::metal
