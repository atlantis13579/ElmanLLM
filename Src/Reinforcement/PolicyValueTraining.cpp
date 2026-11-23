#include "Reinforcement/PolicyValueTraining.h"

#include <cmath>

#include "Reinforcement/Reinforcement.h"

namespace elman::policy {
namespace {
int validate_sample(const Model& model, const Sample& sample) {
  const int actions = validate_observation(model, sample.obs);
  if (sample.policy.size() != size_t(actions) ||
      !(std::abs(sample.value) <= 1)) {
    throw std::invalid_argument("Invalid policy/value target shape or value");
  }
  std::vector<bool> legal(actions);
  for (int a : sample.obs.legal) {
    legal[a] = true;
  }
  if (sample.action < 0 || sample.action >= actions || !legal[sample.action]) {
    throw std::invalid_argument("Sampled action must be legal");
  }
  double mass = 0;
  for (int i = 0; i < actions; ++i) {
    if (!std::isfinite(sample.policy[i]) || sample.policy[i] < 0 ||
        (!legal[i] && sample.policy[i] != 0)) {
      throw std::invalid_argument("Invalid target mass");
    }
    mass += sample.policy[i];
  }
  if (std::abs(mass - 1) > 1e-4) {
    throw std::invalid_argument("Target policy must sum to one");
  }
  return actions;
}
}  // namespace

UpdateResult train_batch(Model& model, Optimizer& opt,
                         const std::vector<Sample>& samples, int batch,
                         Objective objective, std::mt19937& random, bool preselected) {
  if (samples.empty() || batch < 1) {
    throw std::invalid_argument("A training batch needs samples");
  }
  if (preselected && samples.size() != size_t(batch))
    throw std::invalid_argument("A preselected batch needs exactly batch_size samples");
  const Shape& sample_shape = samples.front().obs.shape;
  const int actions = validate_sample(model, samples.front());
  for (size_t i = 1; i < samples.size(); ++i) {
    if (samples[i].obs.shape != sample_shape) {
      throw std::invalid_argument("Mixed shapes in training batch");
    }
    validate_sample(model, samples[i]);
  }
  std::uniform_int_distribution<size_t> draw(0, samples.size() - 1);
  const Backend backend = model.parameters().front().value.backend();
  Shape shape = sample_shape;
  shape.insert(shape.begin(), batch);
  std::vector<float> input, policies, values, mask;
  std::vector<int32_t> chosen;
  for (int i = 0; i < batch; ++i) {
    const policy::Sample& s = samples[preselected ? size_t(i) : draw(random)];
    input.insert(input.end(), s.obs.data.begin(), s.obs.data.end());
    policies.insert(policies.end(), s.policy.begin(), s.policy.end());
    values.push_back(s.value);
    chosen.push_back(s.action);
    if (objective == Objective::Reinforce) {
      size_t start = mask.size();
      mask.resize(start + actions, -1e30f);
      for (int a : s.obs.legal) {
        mask[start + a] = 0;
      }
    }
  }
  for (Parameter& p : model.parameters()) {
    p.value.zero_grad();
  }
  Tensor output = model.forward(Tensor::from_data(shape, input, backend), true);
  Tensor logits = slice(output, 1, 0, actions);
  Tensor targets = Tensor::from_data({batch}, values, backend);
  Tensor cost =
      mse_loss(squeeze(slice(output, 1, actions, actions + 1), 1), targets);
  Tensor policy;
  if (objective == Objective::Reinforce) {
    Tensor masked =
        add(logits, Tensor::from_data({batch, actions}, mask, backend));
    policy = rl::policy_gradient_loss(
        masked, Tensor::from_int_data({batch}, chosen, backend), targets);
  } else if (objective == Objective::Imitation) {
    policy =
        cross_entropy(logits, Tensor::from_int_data({batch}, chosen, backend));
  } else {
    policy = soft_target_cross_entropy(
        logits, Tensor::from_data({batch, actions}, policies, backend));
  }
  Tensor loss = add(policy, cost);
  if (!std::isfinite(loss.data().front())) {
    throw std::runtime_error("Non-finite policy-value loss");
  }
  loss.backward();
  double norm = opt.update(model, opt.learning_rate());
  return {opt.state().step, policy.data().front(), cost.data().front(), norm};
}
}  // namespace elman::policy
