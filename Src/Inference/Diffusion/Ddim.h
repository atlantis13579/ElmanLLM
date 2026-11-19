#pragma once

#include "Core/Tensor.h"
#include "IO/Json.h"

namespace elman {

struct DdimStep {
  Tensor latent;
  Tensor predicted_original;
};

// Deterministic DDIM with leading timestep spacing and epsilon prediction.
class DdimScheduler {
 public:
  DdimScheduler(const Json& configuration, int inference_steps);
  const std::vector<int>& timesteps() const { return timesteps_; }
  DdimStep step(size_t index, const Tensor& noise, const Tensor& latent) const;

 private:
  std::vector<float> alpha_products_;
  std::vector<int> timesteps_;
  int ratio_;
  float final_alpha_, clip_range_;
};

}  // namespace elman
