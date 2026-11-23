#include "Reinforcement/PolicyValueInference.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "Core/TensorDispatch.h"

namespace elman::policy {
namespace {
int action_count(const Model& model, const Shape& shape) {
  if (model.architecture() != "spatial_policy_value" && model.architecture() != "spatial_policy_value_dense") {
    throw std::invalid_argument("Expected a spatial_policy_value model");
  }
  const Shape nominal = model.sample_shape();
  if (nominal.size() != 3 || shape.size() != 3 || shape[0] != nominal[0] ||
      shape[1] <= 0 || shape[2] <= 0) {
    throw std::invalid_argument(
        "Expected positive CHW shape matching model channels");
  }
  if (model.architecture() == "spatial_policy_value_dense" && shape != nominal)
    throw std::invalid_argument("Dense spatial policy-value requires its configured CHW shape");
  int global = model.output_classes() - nominal[1] * nominal[2] - 1;
  int64_t count = int64_t(shape[1]) * shape[2] + global;
  if (count >= std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Action count too large");
  }
  return static_cast<int>(count);
}

}  // namespace

int validate_observation(const Model& model, const Observation& obs) {
  const int count = action_count(model, obs.shape);
  if (obs.data.size() != shape_element_count(obs.shape)) {
    throw std::invalid_argument("Observation length mismatch");
  }
  if (!std::all_of(obs.data.begin(), obs.data.end(),
                   [](float v) { return std::isfinite(v); })) {
    throw std::invalid_argument("Expected finite observation values");
  }
  if (obs.legal.empty()) {
    throw std::invalid_argument("Expected at least one legal action");
  }
  std::vector<bool> seen(count);
  for (int action : obs.legal) {
    if (action < 0 || action >= count || seen[action]) {
      throw std::invalid_argument("Invalid legal action index");
    }
    seen[action] = true;
  }
  return count;
}

Prediction predict(Model& model, const Observation& obs) {
  validate_observation(model, obs);
  Shape shape = obs.shape;
  shape.insert(shape.begin(), 1);
  const Backend backend = model.parameters().front().value.backend();
  Tensor input = Tensor::from_data(shape, obs.data, backend);
  std::vector<float> outputs = model.forward(input, false).data();
  float value = outputs.back();
  outputs.pop_back();
  if (!std::isfinite(value) || std::abs(value) > 1) {
    throw std::runtime_error("Invalid value prediction");
  }
  float largest = -std::numeric_limits<float>::infinity();
  for (int a : obs.legal) {
    if (!std::isfinite(outputs.at(a))) {
      throw std::runtime_error("Non-finite policy logit");
    }
    largest = std::max(largest, outputs[a]);
  }
  double total = 0;
  std::vector<double> policy(outputs.size());
  for (int a : obs.legal) {
    total += (policy[a] = std::exp(double(outputs[a]) - largest));
  }
  for (double& probability : policy) {
    probability /= total;
  }
  return {std::move(policy), value};
}
}  // namespace elman::policy
