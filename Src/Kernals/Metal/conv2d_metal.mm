#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"
#include "Kernals/conv2d_params.h"

namespace elman::metal {
Tensor lower_conv2d(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  Tensor c = inputs[2];
  if (!a && c) {
    return m.spread(m.reshape(c, @[ @1, @(p.output_shape(1)), @1, @1 ]), shape(p));
  }
  if (!a) {
    return m.filled(p, 0);
  }
  using namespace operator_attr::conv2d;
  const int pw = p.attribute(kPaddingWidth), ph = p.attribute(kPaddingHeight);
  MPSGraphConvolution2DOpDescriptor* descriptor =
      [MPSGraphConvolution2DOpDescriptor descriptorWithStrideInX:p.attribute(kStrideWidth)
                                                       strideInY:p.attribute(kStrideHeight)
                                                 dilationRateInX:p.attribute(kDilationWidth)
                                                 dilationRateInY:p.attribute(kDilationHeight)
                                                          groups:p.attribute(kGroups)
                                                     paddingLeft:pw
                                                    paddingRight:pw
                                                      paddingTop:ph
                                                   paddingBottom:ph
                                                    paddingStyle:MPSGraphPaddingStyleExplicit
                                                      dataLayout:MPSGraphTensorNamedDataLayoutNCHW
                                                   weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
  Tensor result = [m.g convolution2DWithSourceTensor:a weightsTensor:b descriptor:descriptor name:nil];
  if (!c) {
    return result;
  }
  return m.add(result, m.reshape(c, @[ @1, @(p.output_shape(operator_attr::conv2d::kChannelAxis)), @1, @1 ]));
}
}  // namespace elman::metal
