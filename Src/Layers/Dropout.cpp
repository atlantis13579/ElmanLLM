#include "Layers/Dropout.h"

#include <cmath>
#include <stdexcept>

namespace elman::nn {

Dropout::Dropout(double probability) : probability_(probability) {
  if (!std::isfinite(probability) || probability < 0 || probability >= 1)
    throw std::invalid_argument("Dropout probability must be in [0,1)");
}

Tensor Dropout::forward(const Tensor& input, uint32_t seed) { return dropout(input, probability_, seed, training()); }

}  // namespace elman::nn
