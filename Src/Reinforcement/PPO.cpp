#include "Reinforcement/PPO.h"
#include <cmath>
#include <stdexcept>

namespace elman::rl {

void generalized_advantages(std::vector<PpoStep>& steps, float bootstrap,
                            double discount, double lambda) {
    if (!(discount >= 0 && discount <= 1 && lambda >= 0 && lambda <= 1) || !std::isfinite(bootstrap))
        throw std::invalid_argument("Invalid PPO discount, lambda or bootstrap");
    double next_value = bootstrap, next_advantage = 0;
    for (size_t i = steps.size(); i-- > 0;) {
        PpoStep& step = steps[i];
        if (!std::isfinite(step.reward) || !std::isfinite(step.value))
            throw std::invalid_argument("Non-finite PPO rollout");
        const double continuation = step.terminal ? 0 : 1;
        const double delta = step.reward + discount * next_value * continuation - step.value;
        next_advantage = delta + discount * lambda * continuation * next_advantage;
        step.advantage = static_cast<float>(next_advantage);
        step.target_value = static_cast<float>(next_advantage + step.value);
        next_value = step.value;
    }
}

Tensor gaussian_log_probability(const Tensor& mean, const Tensor& action, double standard_deviation) {
    if (mean.shape() != action.shape() || mean.shape().size() != 2 ||
        mean.backend() != action.backend() || mean.dtype() != DType::Float32 ||
        action.dtype() != DType::Float32 || !std::isfinite(standard_deviation) || standard_deviation <= 0)
        throw std::invalid_argument("Invalid Gaussian policy tensors or deviation");
    const Tensor difference = div(sub(action, mean), standard_deviation);
    return mul(add(sum(pow(difference, 2), 1),
                   mean.shape()[1] * std::log(2 * 3.14159265358979323846 * standard_deviation * standard_deviation)), -0.5);
}

PpoResult ppo_update(Model& model, Optimizer& optimizer, const std::vector<PpoStep>& batch,
                     double standard_deviation, double clip, double value_coefficient) {
    if (batch.empty() || !(clip > 0 && clip < 1) || !(value_coefficient >= 0))
        throw std::invalid_argument("Invalid PPO minibatch or coefficients");
    const int features = model.sample_shape().at(0);
    const int actions = model.output_classes() - 1;
    if (model.sample_shape().size() != 1 || actions < 1)
        throw std::invalid_argument("PPO expects a flat actor-critic model");
    std::vector<float> observations, actions_data, old_log, advantages, targets;
    for (const PpoStep& step : batch) {
        if (step.observation.size() != size_t(features) || step.action.size() != size_t(actions) ||
            !std::isfinite(step.old_log_probability) || !std::isfinite(step.advantage) ||
            !std::isfinite(step.target_value))
            throw std::invalid_argument("Invalid PPO sample");
        observations.insert(observations.end(), step.observation.begin(), step.observation.end());
        actions_data.insert(actions_data.end(), step.action.begin(), step.action.end());
        old_log.push_back(step.old_log_probability);
        advantages.push_back(step.advantage);
        targets.push_back(step.target_value);
    }
    const int n = static_cast<int>(batch.size());
    const Backend backend = model.parameters().front().value.backend();
    for (Parameter& parameter : model.parameters()) parameter.value.zero_grad();
    Tensor output = model.forward(Tensor::from_data({n, features}, observations, backend), true);
    Tensor mean = slice(output, 1, 0, actions);
    Tensor value = squeeze(slice(output, 1, actions, actions + 1), 1);
    Tensor log_probability = gaussian_log_probability(mean, Tensor::from_data({n, actions}, actions_data, backend), standard_deviation);
    Tensor ratio = exp(sub(log_probability, Tensor::from_data({n}, old_log, backend)));
    Tensor advantage = Tensor::from_data({n}, advantages, backend);
    Tensor clipped = minimum(maximum(ratio, Tensor::from_data({n}, std::vector<float>(n, float(1 - clip)), backend)),
                             Tensor::from_data({n}, std::vector<float>(n, float(1 + clip)), backend));
    Tensor policy_loss = mul(mean_all(minimum(mul(ratio, advantage), mul(clipped, advantage))), -1);
    Tensor value_loss = mean_all(pow(sub(value, Tensor::from_data({n}, targets, backend)), 2));
    Tensor loss = add(policy_loss, mul(value_loss, value_coefficient));
    if (!std::isfinite(loss.data().front())) throw std::runtime_error("Non-finite PPO loss");
    loss.backward();
    const double norm = optimizer.update(model, optimizer.learning_rate());
    return {optimizer.state().step, policy_loss.data().front(), value_loss.data().front(), norm};
}

} // namespace elman::rl
