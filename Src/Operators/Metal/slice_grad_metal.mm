#include <algorithm>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_slice_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  if (!a) {
    return m.filled(p, 0);
  }
  auto source = shape(p);
  auto result = shape(p, 0);
  const int stride = p.attribute(2);
  const int rank = p.output_rank(), axis = p.attribute(0), start = p.attribute(1);
  auto starts = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  auto ends = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  auto strides = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  for (int d = 0; d < rank; ++d) {
    [starts addObject:@(axis == d ? start : 0)];
    [strides addObject:@(axis == d ? stride : 1)];
    const int64_t end = int64_t(start) + int64_t(result[d].intValue) * stride;
    [ends addObject:@(axis == d ? std::min<int64_t>(end, source[d].intValue) : source[d].intValue)];
  }
  return [m.g sliceGradientTensor:a
                 fwdInShapeTensor:m.int_vector(shape(p))
                           starts:starts
                             ends:ends
                          strides:strides
                             name:nil];
}
}  // namespace elman::metal
