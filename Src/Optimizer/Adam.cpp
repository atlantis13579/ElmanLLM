#include <cmath>

#include "Optimizer/OptimizerInternal.h"

namespace elman::optimizer_internal {

void adam_step(std::vector<float>& values, const std::vector<float>& gradients,
               std::vector<float>& first_moment,
               std::vector<float>& second_moment, const StepParameters& step) {
  for (size_t i = 0; i < values.size(); ++i) {
    double gradient =
        gradients[i] * step.gradient_scale + step.weight_decay * values[i];
    first_moment[i] = static_cast<float>(step.beta1 * first_moment[i] +
                                         (1 - step.beta1) * gradient);
    second_moment[i] = static_cast<float>(
        step.beta2 * second_moment[i] + (1 - step.beta2) * gradient * gradient);
    double update =
        (first_moment[i] / step.correction1) /
        (std::sqrt(second_moment[i] / step.correction2) + step.epsilon);
    values[i] = static_cast<float>(values[i] - step.rate * update);
  }
}

}  // namespace elman::optimizer_internal
