#include "Backends/MPSGraph.h"

#include "Operators/attributes.h"
#include "Operators/conv2d_params.h"

namespace elman::metal {
Tensor lower_conv2d_input_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  auto a = inputs[0];
  auto b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  using namespace operator_attr::conv2d;
  const int pw = p.attribute(kPaddingWidth), ph = p.attribute(kPaddingHeight);
  auto* descriptor =
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
  return [m.g convolution2DDataGradientWithIncomingGradientTensor:a
                                                    weightsTensor:b
                                                      outputShape:shape(p)
                                     forwardConvolutionDescriptor:descriptor
                                                             name:nil];
}
}  // namespace elman::metal
