#pragma once

#include <cstdint>

#include "Models/Model.h"

namespace elman {

struct LossScaleState {
  float scale = 1.0f;
  uint32_t finite_steps = 0;
};

class LossScaler {
 public:
  LossScaler(bool enabled, float initial_scale, int growth_interval);
  const LossScaleState& state() const { return state_; }
  void restore(LossScaleState state);
  bool unscale_and_check(Model& model) const;
  void update(bool finite);

 private:
  bool enabled_;
  uint32_t growth_interval_;
  LossScaleState state_;
};

}  // namespace elman
