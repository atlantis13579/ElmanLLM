#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"

namespace elman::metal {
Tensor lower_addmm(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  Tensor c = inputs[2];
  const float alpha = attribute_float(p, operator_attr::addmm::kAlpha);
  const float beta = attribute_float(p, operator_attr::addmm::kBeta);
  if (!a) {
    return c && beta ? m.spread(m.mul(c, m.number(beta, c.dataType)), shape(p)) : m.filled(p, 0);
  }
  const MPSDataType work = is_narrow_float(DType(p.input_dtype(KernelInput::A))) ? MPSDataTypeFloat32 : a.dataType;
  a = m.cast(a, work);
  b = m.cast(b, work);
  if (c) {
    c = m.cast(c, work);
  }
  if (alpha == 0) {
    return beta == 0 ? m.zeros(shape(p), work) : m.spread(m.mul(c, m.number(beta, work)), shape(p));
  }
  Tensor result = m.matmul(a, b);
  if (alpha != 1) {
    result = m.mul(result, m.number(alpha, work));
  }
  if (c && beta) {
    result = m.add(result, beta == 1 ? c : m.mul(c, m.number(beta, work)));
  }
  return result;
}
}  // namespace elman::metal
