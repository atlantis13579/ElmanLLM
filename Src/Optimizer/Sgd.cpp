#include "Optimizer/OptimizerInternal.h"

namespace elman::optimizer_internal {

void sgd_step(std::vector<float>& values, const std::vector<float>& gradients,
              std::vector<float>& first_moment, const StepParameters& step) {
  for (size_t i = 0; i < values.size(); ++i) {
    double gradient =
        gradients[i] * step.gradient_scale + step.weight_decay * values[i];
    first_moment[i] =
        static_cast<float>(step.momentum * first_moment[i] + gradient);
    values[i] = static_cast<float>(values[i] - step.rate * first_moment[i]);
  }
}

}  // namespace elman::optimizer_internal
