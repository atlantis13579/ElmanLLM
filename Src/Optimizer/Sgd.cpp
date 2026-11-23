#include "Optimizer/OptimizerInternal.h"

namespace elman::optimizer_internal {

void sgd_step(std::vector<float>& values, const std::vector<float>& gradients,
              std::vector<float>& first_moment, const FloatOptimizerParameters& step) {
  for (size_t i = 0; i < values.size(); ++i) {
    const FloatOptimizerElement result = float_optimizer_element(
        values[i], gradients[i], first_moment[i], 0.f, step, FloatOptimizerAlgorithm::Sgd, false, i, values.size());
    values[i] = result.value;
    first_moment[i] = result.first;
  }
}

}  // namespace elman::optimizer_internal
