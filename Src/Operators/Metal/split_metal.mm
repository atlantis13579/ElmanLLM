#include <algorithm>

#include "Backends/MPSGraph.h"

namespace elman::metal {
Tensor lower_split(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Shape source = shape(p, 0);
  Shape result = shape(p);
  const int stride = 1;
  const int rank = p.output_rank(), axis = p.attribute(0), start = p.attribute(1);
  NSMutableArray<NSNumber*>* starts = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  NSMutableArray<NSNumber*>* ends = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  NSMutableArray<NSNumber*>* strides = [NSMutableArray<NSNumber*> arrayWithCapacity:rank];
  for (int d = 0; d < rank; ++d) {
    [starts addObject:@(axis == d ? start : 0)];
    [strides addObject:@(axis == d ? stride : 1)];
    const int64_t end = int64_t(start) + int64_t(result[d].intValue) * stride;
    [ends addObject:@(axis == d ? std::min<int64_t>(end, source[d].intValue) : source[d].intValue)];
  }
  return [m.g sliceTensor:a starts:starts ends:ends strides:strides name:nil];
}
}  // namespace elman::metal
