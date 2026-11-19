#include "Inference/Diffusion/Ddim.h"

#include <cmath>
#include <stdexcept>

#include "Core/Device.h"

namespace elman {
namespace {

std::vector<float> betas(const Json& config, int count) {
  if (config.contains("trained_betas") && !config.at("trained_betas").is_null()) {
    std::vector<float> values;
    for (const auto& value : config.at("trained_betas").array()) values.push_back(static_cast<float>(value.number()));
    if (values.size() != size_t(count)) throw std::invalid_argument("DDIM beta count differs from training timesteps");
    return values;
  }
  const auto schedule = config.string_or("beta_schedule", "linear");
  if (schedule != "linear" && schedule != "scaled_linear") throw std::invalid_argument("Unsupported DDIM beta schedule");
  const double start = config.number_or("beta_start", 0.0001);
  const double end = config.number_or("beta_end", 0.02);
  if (!(start > 0 && end >= start && end < 1)) throw std::invalid_argument("Invalid DDIM beta bounds");
  const bool scaled = schedule == "scaled_linear";
  const float first = static_cast<float>(scaled ? std::sqrt(start) : start);
  const float last = static_cast<float>(scaled ? std::sqrt(end) : end);
  const float stride = count > 1 ? (last - first) / (count - 1) : 0.0f;
  std::vector<float> result;
  result.reserve(count);
  for (int i = 0; i < count; ++i) {
    const float value = i < (count + 1) / 2 ? first + i * stride : last - (count - i - 1) * stride;
    result.push_back(scaled ? value * value : value);
  }
  return result;
}

Tensor scale_latent(const Tensor& value, float factor, bool inverse = false) {
  const auto dtype = value.dtype();
  const auto wide = dtype == DType::Float32 ? value : cast(value, DType::Float32);
  auto result = inverse ? div(wide, factor) : mul(wide, factor);
  return result.dtype() == dtype ? result : cast(result, dtype);
}

}  // namespace

DdimScheduler::DdimScheduler(const Json& config, int inference_steps) {
  const int train_steps = config.int_or("num_train_timesteps", 1000);
  const int offset = config.int_or("steps_offset", 0);
  if (train_steps <= 0 || inference_steps <= 0 || inference_steps > train_steps || offset < 0)
    throw std::invalid_argument("Invalid DDIM timestep count or offset");
  if (config.string_or("timestep_spacing", "leading") != "leading" ||
      config.string_or("prediction_type", "epsilon") != "epsilon" ||
      config.bool_or("thresholding", false) || config.bool_or("rescale_betas_zero_snr", false))
    throw std::invalid_argument("DDIM configuration requires leading timesteps and epsilon prediction");
  const bool clip = config.bool_or("clip_sample", true);
  clip_range_ = clip ? static_cast<float>(config.number_or("clip_sample_range", 1.0)) : 0.0f;
  if (clip && (!(clip_range_ > 0.0f) || !std::isfinite(clip_range_)))
    throw std::invalid_argument("Invalid DDIM clipping range");
  ratio_ = train_steps / inference_steps;
  if (int64_t(inference_steps - 1) * ratio_ + offset >= train_steps)
    throw std::invalid_argument("DDIM timesteps exceed the schedule");

  alpha_products_.reserve(train_steps);
  double product = 1;
  for (float beta : betas(config, train_steps)) {
    if (!(beta > 0 && beta < 1)) throw std::invalid_argument("DDIM betas must be in (0,1)");
    product *= 1.0f - beta;
    const float alpha = static_cast<float>(product);
    if (!(alpha > 0)) throw std::invalid_argument("DDIM cumulative alpha underflow");
    alpha_products_.push_back(alpha);
  }
  final_alpha_ = config.bool_or("set_alpha_to_one", true) ? 1.0f : alpha_products_.front();
  timesteps_.reserve(inference_steps);
  for (int i = inference_steps - 1; i >= 0; --i) timesteps_.push_back(i * ratio_ + offset);
}

DdimStep DdimScheduler::step(size_t index, const Tensor& noise, const Tensor& latent) const {
  if (index >= timesteps_.size() || !noise.defined() || !latent.defined() ||
      noise.shape() != latent.shape() || noise.dtype() != latent.dtype() ||
      !is_compute_float(latent.dtype()) || noise.backend() != latent.backend() || noise.device() != latent.device())
    throw std::invalid_argument("Invalid DDIM step tensors or index");
  DeviceScope owner(latent.backend(), latent.device());
  const int timestep = timesteps_[index];
  const int previous = timestep - ratio_;
  const float alpha = alpha_products_[timestep];
  const float previous_alpha = previous >= 0 ? alpha_products_[previous] : final_alpha_;
  auto original = scale_latent(sub(latent, scale_latent(noise, std::sqrt(1.0f - alpha))), std::sqrt(alpha), true);
  if (clip_range_ > 0.0f) {
    auto high = Tensor::from_data({}, {clip_range_}, latent.backend(), false, latent.dtype());
    auto low = Tensor::from_data({}, {-clip_range_}, latent.backend(), false, latent.dtype());
    original = maximum(low, minimum(original, high));
  }
  return {add(scale_latent(original, std::sqrt(previous_alpha)), scale_latent(noise, std::sqrt(1.0f - previous_alpha))), original};
}

}  // namespace elman
