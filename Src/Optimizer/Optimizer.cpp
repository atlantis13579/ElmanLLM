#include "Optimizer/Optimizer.h"

#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Optimizer/OptimizerInternal.h"

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

}  // namespace

Optimizer::Optimizer(const Json& settings, const Model& model) {
  name_ = settings.string_or("optimizer", "adam");
  settings_.learning_rate = settings.number_or("learning_rate", 0.001);
  settings_.beta1 = settings.number_or("beta1", 0.9);
  settings_.beta2 = settings.number_or("beta2", 0.999);
  settings_.epsilon = settings.number_or("epsilon", 1e-8);
  settings_.momentum = settings.number_or("momentum", 0);
  settings_.weight_decay = settings.number_or("weight_decay", 0);
  settings_.clip_norm = settings.number_or("clip_grad_norm", 1);
  require(name_ == "adam" || name_ == "adamw" || name_ == "sgd",
          "optimizer must be adam, adamw or sgd");
  require(settings_.learning_rate > 0 && settings_.epsilon > 0 &&
              settings_.beta1 >= 0 && settings_.beta1 < 1 &&
              settings_.beta2 >= 0 && settings_.beta2 < 1 &&
              settings_.momentum >= 0 && settings_.momentum < 1 &&
              settings_.weight_decay >= 0 && settings_.clip_norm >= 0,
          "Invalid optimizer settings");

  state_.optimizer = name_;
  for (const auto& parameter : model.parameters()) {
    state_.first_moment.emplace_back(parameter.value.numel(), 0.0f);
    state_.second_moment.emplace_back(parameter.value.numel(), 0.0f);
  }
}

double Optimizer::update(Model& model, double rate) {
  require(state_.first_moment.size() == model.parameters().size() &&
              state_.second_moment.size() == model.parameters().size(),
          "Optimizer state does not match model parameters");
  std::vector<std::vector<float>> gradients;
  double norm_squared = 0;
  for (const auto& parameter : model.parameters()) {
    auto gradient = parameter.value.grad();
    require(gradient.size() == parameter.value.numel(),
            "Missing parameter gradient: " + parameter.name);
    for (float value : gradient) {
      require(std::isfinite(value),
              "Non-finite parameter gradient: " + parameter.name);
      norm_squared += static_cast<double>(value) * value;
    }
    gradients.push_back(std::move(gradient));
  }

  double norm = std::sqrt(norm_squared);
  double scale = settings_.clip_norm > 0 && norm > settings_.clip_norm
                     ? settings_.clip_norm / norm
                     : 1;
  ++state_.step;
  optimizer_internal::StepParameters step{
      rate,
      scale,
      settings_.weight_decay,
      settings_.momentum,
      settings_.beta1,
      settings_.beta2,
      settings_.epsilon,
      1 - std::pow(settings_.beta1, static_cast<double>(state_.step)),
      1 - std::pow(settings_.beta2, static_cast<double>(state_.step))};

  for (size_t i = 0; i < model.parameters().size(); ++i) {
    auto& parameter = model.parameters()[i];
    auto values = parameter.value.data();
    auto& first_moment = state_.first_moment[i];
    auto& second_moment = state_.second_moment[i];
    require(first_moment.size() == values.size() &&
                second_moment.size() == values.size(),
            "Optimizer state shape mismatch: " + parameter.name);
    if (name_ == "sgd") {
      optimizer_internal::sgd_step(values, gradients[i], first_moment, step);
    } else if (name_ == "adam") {
      optimizer_internal::adam_step(values, gradients[i], first_moment,
                                    second_moment, step);
    } else {
      optimizer_internal::adamw_step(values, gradients[i], first_moment,
                                     second_moment, step);
    }
    for (float value : values) {
      require(std::isfinite(value),
              "Non-finite optimizer update: " + parameter.name);
    }
    parameter.value.assign(values);
  }
  return norm;
}

}  // namespace elman
