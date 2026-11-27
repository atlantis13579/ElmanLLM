#include "Backends/MPSGraph.h"

#include "Kernals/attributes.h"
#include "Kernals/conv2d_params.h"

namespace elman::metal {
Tensor lower_conv2d_input_grad(Graph m, const Operands& inputs, const KernelParams& p) {
  Tensor a = inputs[0];
  Tensor b = inputs[1];
  if (!a) {
    return m.filled(p, 0);
  }
  using namespace kernal_attr::conv2d;
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
  return [m.g convolution2DDataGradientWithIncomingGradientTensor:a
                                                    weightsTensor:b
                                                      outputShape:shape(p)
                                     forwardConvolutionDescriptor:descriptor
                                                             name:nil];
}
}  // namespace elman::metal
