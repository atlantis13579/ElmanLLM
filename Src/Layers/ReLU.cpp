#include "Layers/ReLU.h"

namespace elman::nn {

Tensor ReLU::forward(const Tensor& input, uint32_t) { return relu(input); }

}  // namespace elman::nn
