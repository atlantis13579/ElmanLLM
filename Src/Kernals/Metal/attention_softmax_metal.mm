#include <algorithm>
#include <cstring>

#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"

namespace elman::metal {
Tensor lower_attention_softmax(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor input = inputs[0];
  Tensor stored = inputs[1];
  const int axis = p.attribute(operator_attr::reduction::kAxis);
  const int rank = input.shape.count, selected = std::max(axis, 0);
  Tensor work = m.float_work(input);
  switch (p.attribute(8)) {
    case 1: {
      Tensor product = m.mul(work, stored);
      return m.mul(work, m.sub(stored, m.sum(product, @[ @(selected) ])));
    }
    case 2:
      return [m.g softMaxWithTensor:work axis:selected name:nil];
    case 3:
    case 5: {
      const int mask = p.attribute(8) == 3 ? 2 : axis;
      NSMutableArray<NSNumber*>* reduced = [NSMutableArray<NSNumber*> array];
      for (int dim = 0; dim < rank; ++dim) {
        if (mask & (1 << dim)) {
          [reduced addObject:@(dim)];
        }
      }
      return reduced.count ? m.sum(work, reduced) : work;
    }
    case 4: {
      if (axis) {
        work = m.transpose(work, rank - 2, rank - 1);
      }
      if (p.attribute(1)) {
        stored = m.transpose(stored, rank - 2, rank - 1);
      }
      return m.matmul(work, stored);
    }
    default: {
      using namespace operator_attr::attention_softmax;
      float divisor = attribute_float(p, kDivisor);
      if (p.attribute(7)) {
        const uint64_t bits = uint64_t(uint32_t(p.attribute(5))) | (uint64_t(uint32_t(p.attribute(6))) << 32);
        double encoded;
        std::memcpy(&encoded, &bits, sizeof(encoded));
        divisor = float(encoded);
      }
      Tensor scaled = m.div(work, m.number(divisor));
      Tensor masked = m.add(m.float_work(m.cast(scaled, input.dataType)), m.float_work(stored));
      work = m.float_work(m.cast(masked, input.dataType));
      break;
    }
  }
  return [m.g softMaxWithTensor:work axis:selected name:nil];
}
}  // namespace elman::metal
