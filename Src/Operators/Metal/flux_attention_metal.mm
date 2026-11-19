#include <cmath>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_flux_attention(Graph m, const Operands& inputs, const KernelParams&) {
  auto query = inputs[0];
  auto key = inputs[1];
  auto value = inputs[2];
  return [m.g scaledDotProductAttentionWithQueryTensor:query
                                             keyTensor:key
                                           valueTensor:value
                                            maskTensor:nil
                                                 scale:1.f / std::sqrt(128.f)
                                                  name:nil];
}
}  // namespace elman::metal
