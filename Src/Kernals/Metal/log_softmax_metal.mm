#include <algorithm>

#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_log_softmax(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  const int selected = std::max(p.attribute(kernal_attr::reduction::kAxis), 0);
  Tensor work = m.float_work(a);
  Tensor shifted = m.sub(work, m.max(work, @[ @(selected) ]));
  return m.sub(shifted, m.log(m.sum(m.exp(shifted), @[ @(selected) ])));
}
}  // namespace elman::metal
