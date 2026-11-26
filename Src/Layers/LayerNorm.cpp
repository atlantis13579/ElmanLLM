#include "Layers/LayerNorm.h"
#include "Kernals/inference_ops.h"

namespace elman::inference {

Tensor LayerNorm::operator()(const Tensor& value) const { return layer_norm(value, weight, bias, epsilon); }

}  // namespace elman::inference
