#pragma once

#include "Layers/Module.h"

namespace elman::nn {

class ReLU final : public Module {
  Tensor forward(const Tensor& input, uint32_t) override;
};

}  // namespace elman::nn
