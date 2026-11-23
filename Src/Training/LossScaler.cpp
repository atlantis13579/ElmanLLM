#include "Training/LossScaler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

#include "Core/Device.h"
#include "Core/Graph.h"

namespace elman {
namespace {

bool finite_gradient(const Tensor& gradient) {
  if (!gradient.numel()) return true;
  const int count = static_cast<int>(gradient.numel());
  const Tensor magnitudes = abs(gradient);
  constexpr int chunk = 4096;
  const int rows = count / chunk;
  const int tail = count % chunk;
  if (rows && !std::isfinite(amax(amax(reshape(contiguous_view(magnitudes, {rows * chunk}), {rows, chunk}), 1), 0).data().front()))
    return false;
  return !tail || std::isfinite(amax(contiguous_view(magnitudes, {tail}, size_t(rows) * chunk), 0).data().front());
}

}  // namespace

LossScaler::LossScaler(bool enabled, float initial_scale, int growth_interval)
    : enabled_(enabled), growth_interval_(static_cast<uint32_t>(growth_interval)) {
  if (!std::isfinite(initial_scale) || initial_scale < std::numeric_limits<float>::min() || growth_interval <= 0)
    throw std::invalid_argument("Invalid FP16 loss scale or growth interval");
  if (enabled_) state_.scale = initial_scale;
}

void LossScaler::restore(LossScaleState state) {
  if (!std::isfinite(state.scale) || state.scale < std::numeric_limits<float>::min() ||
      state.finite_steps >= growth_interval_ || (!enabled_ && (state.scale != 1.f || state.finite_steps != 0)))
    throw std::invalid_argument("Invalid checkpoint loss scaling state");
  state_ = state;
}

bool LossScaler::unscale_and_check(Model& model) const {
  if (!enabled_) return true;
  graph::RecordingGuard no_grad;
  bool finite = true;
  for (Parameter& parameter : model.parameters()) {
    Tensor& value = parameter.value;
    if (!value.node()->gradient) continue;
    if (value.dtype() != DType::Float32) throw std::invalid_argument("FP16 scaling requires Float32 parameters");
    DeviceScope owner(value.backend(), value.device());
    Tensor grad = Tensor(value.node()->gradient);
    Tensor unscaled = state_.scale == 1.f ? grad : mul(grad, 1.0 / state_.scale);
    finite &= finite_gradient(unscaled);
    value.node()->gradient = unscaled.node();
  }
  return finite;
}

void LossScaler::update(bool finite) {
  if (!enabled_) return;
  if (!finite) {
    state_.scale = std::max(state_.scale * 0.5f, std::numeric_limits<float>::min());
    state_.finite_steps = 0;
  } else if (++state_.finite_steps == growth_interval_) {
    if (state_.scale <= std::numeric_limits<float>::max() * 0.5f) state_.scale *= 2.0f;
    state_.finite_steps = 0;
  }
}

}  // namespace elman
