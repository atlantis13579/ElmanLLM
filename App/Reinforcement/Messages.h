#pragma once
#include "Reinforcement/PolicyValueTraining.h"

namespace elman::policy {
Observation read_observation(const Json& message);
Sample read_sample(const Json& message);
Objective read_objective(const Json& message);
Json::Object prediction_message(const Prediction& prediction);
Json loss_message(const UpdateResult& update);
}  // namespace elman::policy
