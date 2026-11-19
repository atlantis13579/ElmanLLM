#include "Inference/Diffusion/Flux2Sampler.h"

#include <cmath>
#include <stdexcept>

#include "Core/Device.h"
#include "Operators/flux2_ops.h"
#include "Operators/inference_ops.h"

namespace elman::flux2 {
std::vector<float> flow_schedule(int steps, int image_tokens) {
  if (steps < 1 || image_tokens < 1) {
    throw std::invalid_argument("FLUX.2 needs positive step and image token counts");
  }
  const double at200 = .00016927 * image_tokens + .45666666;
  const double at10 = 8.73809524e-5 * image_tokens + 1.89833333;
  const double mu = image_tokens > 4300 ? at200 : at10 + (steps - 10.) * (at200 - at10) / 190.;
  std::vector<float> result(size_t(steps) + 1);
  result.front() = 1.f;
  result.back() = 0.f;
  for (int i = 1; i < steps; ++i) {
    result[i] = float(1. / (1. + std::exp(-mu) * double(i) / (steps - i)));
  }
  return result;
}

Tensor flow_step(const Tensor& latent, const Tensor& velocity, float current, float next) {
  if (latent.shape() != velocity.shape() || !is_compute_float(latent.dtype()) || velocity.dtype() != latent.dtype() ||
      !(0.f <= next && next <= current && current <= 1.f)) {
    throw std::invalid_argument("Invalid flow matching Euler update");
  }
  DeviceScope owner(latent.backend(), latent.device());
  return add(latent, mul(velocity, next - current));
}

Tensor guided_velocity(const Tensor& unconditional, const Tensor& conditional, float guidance) {
  if (unconditional.shape() != conditional.shape() || unconditional.dtype() != conditional.dtype() ||
      !is_compute_float(unconditional.dtype()) || !std::isfinite(guidance)) {
    throw std::invalid_argument("Invalid flow matching guidance");
  }
  DeviceScope owner(unconditional.backend(), unconditional.device());
  return add(unconditional, mul(sub(conditional, unconditional), guidance));
}

Tensor image_positions(int height, int width, int index, Backend backend) {
  if (height <= 0 || width <= 0 || int64_t(height) * width > INT32_MAX / 4 || index < 0 || index > INT32_MAX / 10) {
    throw std::invalid_argument("Invalid FLUX.2 image lattice");
  }
  const int length = height * width;
  std::vector<int32_t> ids(size_t(length) * 4);
  for (int i = 0; i < length; ++i) {
    ids[4 * i] = 10 * index;
    ids[4 * i + 1] = i / width;
    ids[4 * i + 2] = i % width;
  }
  return Tensor::from_int_data({1, length, 4}, ids, backend);
}
Tensor text_positions(int length, Backend backend) {
  if (length < 1 || length > INT32_MAX / 4) {
    throw std::invalid_argument("Invalid FLUX.2 text length");
  }
  std::vector<int32_t> ids(size_t(length) * 4);
  for (int i = 0; i < length; ++i) {
    ids[4 * i + 3] = i;
  }
  return Tensor::from_int_data({1, length, 4}, ids, backend);
}

Tensor sample(const Transformer& model, const Tensor& noise, const Tensor& prompt, const Tensor& empty_prompt,
              const std::vector<Tensor>& references, const SamplingOptions& options) {
  const auto& s = noise.shape();
  if (s.size() != 4 || s[0] < 1 || s[1] != 128 || prompt.shape().size() != 3 || prompt.shape()[0] != s[0]) {
    throw std::invalid_argument("Invalid FLUX.2 noise or prompt batch");
  }
  const auto variant = model.variant();
  const bool cfg = variant == Variant::Base4B || variant == Variant::Base9B, dev = variant == Variant::Dev;
  if (cfg && (!empty_prompt.defined() || empty_prompt.shape() != prompt.shape())) {
    throw std::invalid_argument("Base checkpoint CFG requires empty-prompt conditioning for each image");
  }
  const int steps = options.steps.value_or(cfg || dev ? 50 : 4);
  const float guidance = options.guidance.value_or(cfg || dev ? 4.f : 1.f);
  if (!std::isfinite(guidance) || (!cfg && !dev && guidance != 1.f)) {
    throw std::invalid_argument("Invalid guidance for selected FLUX.2 checkpoint");
  }

  DeviceScope owner(noise.backend(), noise.device());
  auto img = image_tokens(noise);
  auto grid = [&](int h, int w, int index) { return rotary_table(image_positions(h, w, index, noise.backend())); };
  auto ip = grid(s[2], s[3], 0), tp = rotary_table(text_positions(prompt.shape()[1], noise.backend()));
  std::vector<Tensor> tokens, tables;
  for (const auto& ref : references) {
    const auto& r = ref.shape();
    if (r.size() != 4 || r[0] != s[0] || r[1] != 128 || ref.dtype() != noise.dtype()) {
      throw std::invalid_argument("Invalid FLUX.2 reference latent");
    }
    tokens.push_back(image_tokens(ref));
    tables.push_back(grid(r[2], r[3], int(tokens.size())));
  }
  Tensor ref, rp;
  if (!tokens.empty()) {
    ref = concat(tokens, 1);
    rp = concat(tables, 1);
  }
  const auto schedule = flow_schedule(steps, img.shape()[1]);
  auto scalar_batch = [&](float x) {
    return Tensor::from_data({s[0]}, std::vector<float>(s[0], x), noise.backend(), false, noise.dtype());
  };
  Tensor g = dev ? scalar_batch(guidance) : Tensor{};
  const bool cached = variant == Variant::Klein9BKV && ref.defined();
  ReferenceCache cache;
  for (int i = 0; i < steps; ++i) {
    if (options.before_step) options.before_step(i + 1);
    auto time = scalar_batch(schedule[i]);
    const bool omit = cached && i > 0;
    auto velocity = model.predict(img, prompt, time, ip, tp, g, omit ? Tensor{} : ref, omit ? Tensor{} : rp,
                                  cached ? &cache : nullptr);
    if (cfg) {
      velocity = guided_velocity(model.predict(img, empty_prompt, time, ip, tp, {}, ref, rp), velocity, guidance);
    }
    img = flow_step(img, velocity, schedule[i], schedule[i + 1]);
    if (options.on_step) {
      options.on_step(i + 1, token_image(img, s[2], s[3]));
    }
  }
  return token_image(img, s[2], s[3]);
}
}  // namespace elman::flux2
