#pragma once
#include <random>

#include "Reinforcement/PolicyValueInference.h"
#include "Optimizer/Optimizer.h"

namespace elman::policy {
enum class Objective { Visits, Imitation, Reinforce };
struct Sample {
  Observation obs;
  std::vector<float> policy;
  float value;
  int action;
};
struct UpdateResult {
  uint64_t step;
  float policy_loss, value_loss;
  double gradient_norm;
};

UpdateResult train_batch(Model& model, Optimizer& optimizer,
                         const std::vector<Sample>& samples, int batch_size,
                         Objective objective, std::mt19937& random, bool preselected = false);
}  // namespace elman::policy
