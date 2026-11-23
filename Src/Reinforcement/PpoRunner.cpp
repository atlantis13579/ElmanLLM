#include "Reinforcement/PpoRunner.h"
#include <algorithm>
#include <cmath>
#include <future>
#include <iterator>
#include <numeric>
#include <stdexcept>

namespace elman::rl {

PpoRunner::PpoRunner(Model& model, Optimizer& optimizer, std::vector<PpoEnvironment*> environments,
                     PpoInferenceFactory inference_factory, PpoRunnerSettings settings)
    : model_(model), optimizer_(optimizer), environments_(std::move(environments)),
      inference_factory_(std::move(inference_factory)), settings_(settings), random_(settings.seed) {
    if (environments_.empty() || !inference_factory_ || settings_.rollout_actions < int(environments_.size()) ||
        settings_.rollout_actions % environments_.size() != 0 || settings_.minibatch_size < 1 ||
        settings_.epochs < 1 || !(settings_.standard_deviation > 0) || !(settings_.clip > 0 && settings_.clip < 1))
        throw std::invalid_argument("Invalid PPO runner settings");
    for (PpoEnvironment* environment : environments_)
        if (!environment) throw std::invalid_argument("Null PPO environment");
}

PpoIteration PpoRunner::update() {
    const PpoInference inference = inference_factory_(model_);
    if (!inference) throw std::invalid_argument("Empty PPO inference policy");
    const size_t count = environments_.size();
    const size_t actions = size_t(model_.output_classes() - 1);
    std::normal_distribution<float> noise(0, settings_.standard_deviation);
    std::vector<std::vector<PpoStep>> tracks(count);
    std::vector<PpoStep> rollout;
    rollout.reserve(settings_.rollout_actions);
    float reward_sum = 0;
    int episodes = 0;
    for (int t = 0; t < settings_.rollout_actions / int(count); ++t) {
        std::vector<PpoStep> steps(count);
        std::vector<std::future<PpoOutcome>> jobs;
        jobs.reserve(count);
        for (size_t e = 0; e < count; ++e) {
            PpoStep& step = steps[e];
            step.observation = environments_[e]->observe();
            const std::vector<float> output = inference(step.observation);
            if (output.size() != actions + 1) throw std::invalid_argument("Invalid PPO inference output");
            step.value = output[actions];
            step.action.resize(actions);
            step.old_log_probability = -0.5f * float(actions) *
                std::log(2 * 3.14159265358979323846f * settings_.standard_deviation * settings_.standard_deviation);
            for (size_t a = 0; a < actions; ++a) {
                const float perturbation = noise(random_);
                step.action[a] = output[a] + perturbation;
                step.old_log_probability -= 0.5f * perturbation * perturbation /
                    (settings_.standard_deviation * settings_.standard_deviation);
            }
            if (count > 1)
                jobs.push_back(std::async(std::launch::async, [&, e] { return environments_[e]->step(steps[e].action); }));
        }
        for (size_t e = 0; e < count; ++e) {
            const PpoOutcome outcome = count == 1
                ? environments_[e]->step(steps[e].action) : jobs[e].get();
            steps[e].reward = settings_.reward_scale * outcome.reward;
            steps[e].terminal = outcome.terminal;
            reward_sum += outcome.reward;
            tracks[e].push_back(std::move(steps[e]));
            if (settings_.on_environment_step) settings_.on_environment_step(e, *environments_[e]);
            if (outcome.terminal) { ++episodes; environments_[e]->reset(); }
        }
    }
    for (size_t e = 0; e < count; ++e) {
        const float bootstrap = inference(environments_[e]->observe())[actions];
        generalized_advantages(tracks[e], bootstrap);
        rollout.insert(rollout.end(), std::make_move_iterator(tracks[e].begin()),
                       std::make_move_iterator(tracks[e].end()));
    }
    float mean = 0, square = 0;
    for (const PpoStep& step : rollout) mean += step.advantage;
    mean /= rollout.size();
    for (const PpoStep& step : rollout) square += (step.advantage - mean) * (step.advantage - mean);
    const float scale = std::sqrt(square / rollout.size() + 1e-8f);
    for (PpoStep& step : rollout) step.advantage = (step.advantage - mean) / scale;
    std::vector<int> order(rollout.size());
    std::iota(order.begin(), order.end(), 0);
    PpoIteration result;
    result.update = ++updates_;
    result.mean_reward = reward_sum / rollout.size();
    result.episodes = episodes;
    int batches = 0;
    for (int epoch = 0; epoch < settings_.epochs; ++epoch) {
        std::shuffle(order.begin(), order.end(), random_);
        for (size_t begin = 0; begin < order.size(); begin += settings_.minibatch_size) {
            std::vector<PpoStep> batch;
            for (size_t i = begin; i < std::min(begin + settings_.minibatch_size, order.size()); ++i)
                batch.push_back(rollout[order[i]]);
            const PpoResult updated = ppo_update(model_, optimizer_, batch,
                settings_.standard_deviation, settings_.clip, settings_.value_coefficient);
            result.policy_loss += updated.policy_loss;
            result.value_loss += updated.value_loss;
            result.gradient_norm += updated.gradient_norm;
            ++batches;
        }
    }
    result.policy_loss /= batches;
    result.value_loss /= batches;
    result.gradient_norm /= batches;
    return result;
}

void PpoRunner::run(int updates, const std::function<void(const PpoIteration&)>& on_update,
                    const std::function<bool()>& should_stop) {
    if (updates < 0) throw std::invalid_argument("Invalid PPO update count");
    for (int i = 0; i < updates; ++i) {
        if (should_stop && should_stop()) break;
        const PpoIteration result = update();
        if (on_update) on_update(result);
    }
}

} // namespace elman::rl
