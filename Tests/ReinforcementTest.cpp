#include "Reinforcement/Reinforcement.h"
#include "Reinforcement/PolicyValueTraining.h"
#include "Reinforcement/PPO.h"
#include <cmath>
#include <iostream>

using namespace elman;
namespace {
void policy_value_updates() {
    const Json definition = Json::parse(R"({"format":"elman.model","version":1,"architecture":"spatial_policy_value",
        "name":"fixture","input_shape":[2,2,3],"channels":4,"blocks":1,"global_actions":2})");
    const policy::Observation obs{{2, 2, 3}, {0, 1, .5f, 0, 1, .5f, 0, 1, .5f, 0, 1, .5f}, {1, 4, 7}};
    const policy::Sample sample{obs, {0, .1f, 0, 0, .9f, 0, 0, 0}, -1, 4};
    for (elman::policy::Objective objective : {policy::Objective::Visits, policy::Objective::Reinforce, policy::Objective::Imitation}) {
        std::unique_ptr<Model> model = create_model(definition, Backend::CPU, 7);
        Optimizer optimizer(Json::Object{{"optimizer", "adam"}, {"learning_rate", .001}}, *model);
        std::mt19937 rng(17);
        policy::Prediction before = policy::predict(*model, obs);
        policy::UpdateResult update = policy::train_batch(*model, optimizer, {sample}, 2, objective, rng);
        policy::Prediction after = policy::predict(*model, obs);
        if (update.step != 1 || after.policy == before.policy || after.value == before.value ||
            !(std::abs(after.value) <= 1) || after.policy.size() != 8)
            throw std::runtime_error("Policy and value heads must both update");
        double total = 0;
        for (size_t i = 0; i < after.policy.size(); ++i) {
            if ((after.policy[i] > 0) != (i == 1 || i == 4 || i == 7))
                throw std::runtime_error("Invalid masked prediction");
            total += after.policy[i];
        }
        if (std::abs(total - 1) > 1e-6) throw std::runtime_error("Invalid prediction mass");
        const std::mt19937 before_batch = rng;
        policy::train_batch(*model, optimizer, {sample, sample}, 2, objective, rng, true);
        if (rng != before_batch) throw std::runtime_error("A preselected minibatch was sampled twice");
    }
}

void continuous_ppo_updates() {
    const Json definition = Json::parse(R"({"format":"elman.model","version":1,"architecture":"continuous_actor_critic",
        "name":"continuous policy","input_features":5,"actions":2,"hidden_features":8,"hidden_layers":2})");
    std::unique_ptr<Model> model = create_model(definition, Backend::CPU, 7);
    if (model->parameters().size() != 6 || model->output_classes() != 3)
        throw std::runtime_error("Continuous actor-critic dimensions are incorrect");
    Optimizer optimizer(Json::Object{{"optimizer", "adam"}, {"learning_rate", .001}}, *model);
    rl::PpoStep sample;
    sample.observation.assign(5, 0.1f);
    sample.action.assign(2, 0.2f);
    Tensor initial = model->forward(Tensor::from_data({1, 5}, sample.observation), false);
    sample.old_log_probability = rl::gaussian_log_probability(
        slice(initial, 1, 0, 2), Tensor::from_data({1, 2}, sample.action), 0.35).data()[0];
    sample.reward = 1;
    std::vector<rl::PpoStep> rollout{sample};
    rl::generalized_advantages(rollout, 0);
    if (std::abs(rollout[0].advantage - 1) > 1e-6f ||
        std::abs(rollout[0].target_value - 1) > 1e-6f)
        throw std::runtime_error("Incorrect PPO advantages");
    rl::PpoResult result = rl::ppo_update(*model, optimizer, rollout);
    Tensor updated = model->forward(Tensor::from_data({1, 5}, sample.observation), false);
    if (result.step != 1 || updated.data() == initial.data() || !std::isfinite(result.policy_loss) ||
        !std::isfinite(result.value_loss))
        throw std::runtime_error("PPO did not update the continuous actor and critic");
}
} // namespace

int main() {
    try {
        if (rl::discounted_returns({1, -1}, 0.5, 2) != std::vector<float>{1, 0})
            throw std::runtime_error("Incorrect bootstrapped returns");
        rl::ReplayBuffer<int> replay(2);
        for (int i : {1, 2, 3}) replay.push(i);
        std::mt19937 rng(17);
        for (int i = 0; i < 20; ++i)
            if (replay.draw(rng) < 2) throw std::runtime_error("Expired replay sample");

        Tensor logits = Tensor::zeros({2, 2}, Backend::CPU, true);
        Tensor reward = Tensor::from_data({2}, {1, -1}, Backend::CPU, true);
        Tensor loss = rl::policy_gradient_loss(logits, Tensor::from_int64_data({2}, {1, 0}), reward);
        loss.backward();
        if (std::abs(loss.data()[0]) > 1e-6f || logits.grad() != std::vector<float>{.25f, -.25f, .25f, -.25f})
            throw std::runtime_error("Incorrect signed policy gradient");
        if (!reward.grad().empty()) throw std::runtime_error("Advantage must be detached");
        logits.zero_grad();
        Tensor soft = soft_target_cross_entropy(logits, Tensor::from_data({2, 2}, {.25f, .75f, 1, 0}));
        soft.backward();
        if (std::abs(soft.data()[0] - std::log(2.f)) > 1e-6f ||
            logits.grad() != std::vector<float>{.125f, -.125f, -.25f, .25f})
            throw std::runtime_error("Incorrect soft target loss or gradient");
        policy_value_updates();
        continuous_ppo_updates();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
