#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"
#include "Kernals/conv2d_params.h"

namespace elman::metal {
Tensor lower_max_pool2d(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  using namespace kernal_attr::pool2d;
  const int px = p.attribute(kPaddingWidth), py = p.attribute(kPaddingHeight);
  MPSGraphPooling2DOpDescriptor* descriptor =
      [MPSGraphPooling2DOpDescriptor descriptorWithKernelWidth:p.attribute(kKernelWidth)
                                                  kernelHeight:p.attribute(kKernelHeight)
                                                     strideInX:p.attribute(kStrideWidth)
                                                     strideInY:p.attribute(kStrideHeight)
                                               dilationRateInX:1
                                               dilationRateInY:1
                                                   paddingLeft:px
                                                  paddingRight:px
                                                    paddingTop:py
                                                 paddingBottom:py
                                                  paddingStyle:MPSGraphPaddingStyleExplicit
                                                    dataLayout:MPSGraphTensorNamedDataLayoutNCHW];
  descriptor.includeZeroPadToAverage = p.attribute(kCountIncludePadding) != 0;
  return [m.g maxPooling2DWithSourceTensor:a descriptor:descriptor name:nil];
}
}  // namespace elman::metal
