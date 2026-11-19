#pragma once

#include <optional>

#include "Models/ComponentLayers.h"

namespace elman::diffusion {

struct ResnetBlock {
  inference::GroupNorm first_norm, second_norm;
  inference::Conv2d first, second;
  std::optional<inference::Linear> time;
  std::optional<inference::Conv2d> shortcut;
  ResnetBlock(const io::ComponentWeights& weights, const std::string& prefix, int input, int output,
              int groups, double epsilon, int time_width = 0, const char* shortcut_name = ".conv_shortcut");
  Tensor operator()(const Tensor& input, const Tensor& time_embedding = {}) const;
};

struct SpatialTransformer {
  inference::GroupNorm group_norm;
  inference::Conv2d project_in, project_out;
  inference::LayerNorm self_norm, cross_norm, ff_norm;
  inference::Attention self_attention, cross_attention;
  inference::Linear ff_gate, ff_out;
  int widened;
  SpatialTransformer(const io::ComponentWeights& weights, const std::string& prefix, int width,
                     int context, int heads, int groups);
  Tensor operator()(const Tensor& input, const Tensor& context) const;
};

}  // namespace elman::diffusion
