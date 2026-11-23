#pragma once

#include "Reinforcement/PPO.h"
#include <functional>
#include <random>

namespace elman::rl {

struct PpoOutcome {
    float reward = 0;
    bool terminal = false;
};

class PpoEnvironment {
public:
    virtual ~PpoEnvironment() = default;
    virtual std::vector<float> observe() const = 0;
    virtual PpoOutcome step(const std::vector<float>& action) = 0;
    virtual void reset() = 0;
};

using PpoInference = std::function<std::vector<float>(const std::vector<float>&)>;
using PpoInferenceFactory = std::function<PpoInference(const Model&)>;

struct PpoRunnerSettings {
    int rollout_actions = 1024;
    int minibatch_size = 64;
    int epochs = 1;
    float standard_deviation = 0.15f;
    float reward_scale = 0.02f;
    double clip = 0.2;
    double value_coefficient = 0.1;
    uint32_t seed = 42;
    std::function<void(size_t, const PpoEnvironment&)> on_environment_step;
};

struct PpoIteration {
    int update = 0;
    float mean_reward = 0;
    int episodes = 0;
    double policy_loss = 0, value_loss = 0, gradient_norm = 0;
};

class PpoRunner {
public:
    PpoRunner(Model& model, Optimizer& optimizer, std::vector<PpoEnvironment*> environments,
              PpoInferenceFactory inference_factory, PpoRunnerSettings settings = {});
    PpoIteration update();
    void run(int updates, const std::function<void(const PpoIteration&)>& on_update = {},
             const std::function<bool()>& should_stop = {});

private:
    Model& model_;
    Optimizer& optimizer_;
    std::vector<PpoEnvironment*> environments_;
    PpoInferenceFactory inference_factory_;
    PpoRunnerSettings settings_;
    std::mt19937 random_;
    int updates_ = 0;
};

} // namespace elman::rl
