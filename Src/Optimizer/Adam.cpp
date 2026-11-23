#include "Optimizer/OptimizerInternal.h"

namespace elman::optimizer_internal {

void adam_step(std::vector<float>& values, const std::vector<float>& gradients,
               std::vector<float>& first_moment,
               std::vector<float>& second_moment, const FloatOptimizerParameters& step) {
  for (size_t i = 0; i < values.size(); ++i) {
    const FloatOptimizerElement result = float_optimizer_element(
        values[i], gradients[i], first_moment[i], second_moment[i], step, FloatOptimizerAlgorithm::Adam, false, i, values.size());
    values[i] = result.value;
    first_moment[i] = result.first;
    second_moment[i] = result.second;
  }
}

}  // namespace elman::optimizer_internal
