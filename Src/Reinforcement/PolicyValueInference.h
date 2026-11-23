#pragma once
#include "Models/Model.h"

namespace elman::policy {
struct Observation {
  Shape shape;
  std::vector<float> data;
  std::vector<int> legal;
};
struct Prediction {
  std::vector<double> policy;
  float value;
};

// Returns the number of categorical actions for the supplied spatial shape.
int validate_observation(const Model& model, const Observation& observation);
Prediction predict(Model& model, const Observation& observation);
}  // namespace elman::policy
