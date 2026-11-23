#pragma once

#include "Models/Model.h"
#include "Optimizer/Optimizer.h"

namespace elman::rl {

struct PpoStep {
    std::vector<float> observation, action;
    float old_log_probability = 0;
    float value = 0;
    float reward = 0;
    bool terminal = false;
    float advantage = 0, target_value = 0;
};

// Bootstrap is the critic value of the observation after the final step.
void generalized_advantages(std::vector<PpoStep>& steps, float bootstrap,
                            double discount = 0.99, double lambda = 0.95);

Tensor gaussian_log_probability(const Tensor& mean, const Tensor& action, double standard_deviation);

struct PpoResult {
    uint64_t step;
    float policy_loss, value_loss;
    double gradient_norm;
};

PpoResult ppo_update(Model& model, Optimizer& optimizer, const std::vector<PpoStep>& batch,
                     double standard_deviation = 0.35, double clip = 0.2,
                     double value_coefficient = 0.5);

} // namespace elman::rl
