#pragma once

#include <cstddef>
#include <vector>

namespace elman::optimizer_internal {

struct StepParameters {
  double rate;
  double gradient_scale;
  double weight_decay;
  double momentum;
  double beta1;
  double beta2;
  double epsilon;
  double correction1;
  double correction2;
};

void sgd_step(std::vector<float>& values, const std::vector<float>& gradients,
              std::vector<float>& first_moment, const StepParameters& step);
void adam_step(std::vector<float>& values, const std::vector<float>& gradients,
               std::vector<float>& first_moment,
               std::vector<float>& second_moment, const StepParameters& step);
void adamw_step(std::vector<float>& values, const std::vector<float>& gradients,
                std::vector<float>& first_moment,
                std::vector<float>& second_moment, const StepParameters& step);

}  // namespace elman::optimizer_internal
