#pragma once

#include <optional>

#include "Models/Diffusion/Flux2Transformer.h"

namespace elman::flux2 {
std::vector<float> flow_schedule(int steps, int image_tokens);
Tensor flow_step(const Tensor& latent, const Tensor& velocity, float current, float next);
Tensor guided_velocity(const Tensor& unconditional, const Tensor& conditional, float guidance);
// A zero image index describes the target. References use one-based indices.
Tensor image_positions(int height, int width, int image_index, Backend backend);
Tensor text_positions(int length, Backend backend);

struct SamplingOptions {
  // Omitted values select the checkpoint defaults.
  std::optional<int> steps;
  std::optional<float> guidance;
  std::function<void(int completed_steps, const Tensor& packed_latents)> on_step;
  std::function<void(int step)> before_step;
};
// All image tensors are packed NCHW. The caller supplies the initial noise
// and encoded prompt(s). The result can be passed directly to the VAE decoder.
Tensor sample(const Transformer& model, const Tensor& noise, const Tensor& prompt, const Tensor& empty_prompt = {},
              const std::vector<Tensor>& references = {}, const SamplingOptions& options = {});
}  // namespace elman::flux2
