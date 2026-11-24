#include "Messages.h"

namespace elman::policy {
namespace {
std::vector<float> numbers(const Json& array) {
  std::vector<float> result;
  result.reserve(array.array().size());
  for (const Json& item : array.array()) {
    result.push_back(static_cast<float>(item.number()));
  }
  return result;
}
}  // namespace
Observation read_observation(const Json& message) {
  Observation obs;
  for (const Json& dim : message.at("shape").array()) {
    obs.shape.push_back(dim.integer());
  }
  obs.data = numbers(message.at("observation"));
  for (const Json& entry : message.at("legal").array()) {
    obs.legal.push_back(entry.integer());
  }
  return obs;
}
Sample read_sample(const Json& row) {
  return {read_observation(row), numbers(row.at("policy")),
          static_cast<float>(row.at("value").number()),
          row.at("action").integer()};
}
Objective read_objective(const Json& message) {
  const std::string& name = message.at("objective").string();
  if (name == "visits") {
    return Objective::Visits;
  }
  if (name == "imitation") {
    return Objective::Imitation;
  }
  if (name == "reinforce") {
    return Objective::Reinforce;
  }
  throw std::invalid_argument("Unknown policy objective");
}
Json::Object prediction_message(const Prediction& prediction) {
  Json::Array probabilities;
  probabilities.reserve(prediction.policy.size());
  for (double p : prediction.policy) {
    probabilities.emplace_back(p);
  }
  return {{"policy", std::move(probabilities)}, {"value", prediction.value}};
}
Json loss_message(const UpdateResult& update) {
  return Json::Object{{"step", double(update.step)},
                      {"policy_loss", update.policy_loss},
                      {"value_loss", update.value_loss},
                      {"gradient_norm", update.gradient_norm}};
}
}  // namespace elman::policy
