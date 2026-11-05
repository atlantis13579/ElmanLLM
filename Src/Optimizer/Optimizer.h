#pragma once

#include <string>

#include "IO/Checkpoint.h"

namespace elman {

struct OptimizerSettings {
  double learning_rate = 0.001;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  double momentum = 0;
  double weight_decay = 0;
  double clip_norm = 1;
};

class Optimizer {
 public:
  Optimizer(const Json& settings, const Model& model);

  const std::string& name() const {
    return name_;
  }
  double learning_rate() const {
    return settings_.learning_rate;
  }
  TrainingState& state() {
    return state_;
  }
  const TrainingState& state() const {
    return state_;
  }

  double update(Model& model, double rate);

 private:
  std::string name_;
  OptimizerSettings settings_;
  TrainingState state_;
};

}  // namespace elman
