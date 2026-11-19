#pragma once

#include <stdexcept>

#include "IO/ComponentWeights.h"
#include "NN/InferenceLayers.h"

namespace elman::model_weights {

inline inference::Linear linear(const io::ComponentWeights& weights, const std::string& prefix,
                                int input, int output, bool use_bias = true, bool pointwise = false) {
  return {pointwise ? weights.matrix(prefix + ".weight", output, input)
                    : weights.get(prefix + ".weight", {output, input}),
          use_bias ? weights.get(prefix + ".bias", {output}) : Tensor{}};
}

inline inference::LayerNorm layer_norm(const io::ComponentWeights& weights, const std::string& prefix,
                                       int width, double epsilon) {
  return {weights.get(prefix + ".weight", {width}), weights.get(prefix + ".bias", {width}), epsilon};
}

inline inference::Conv2d conv2d(const io::ComponentWeights& weights, const std::string& prefix,
                               int input, int output, int kernel, int stride = 1, int padding = 0) {
  return {weights.get(prefix + ".weight", {output, input, kernel, kernel}),
          weights.get(prefix + ".bias", {output}), stride, padding};
}

inline inference::GroupNorm group_norm(const io::ComponentWeights& weights, const std::string& prefix,
                                       int channels, int groups, double epsilon) {
  if (groups <= 0 || channels % groups || !(epsilon > 0))
    throw std::invalid_argument("Invalid checkpoint GroupNorm layout for " + prefix);
  return {weights.get(prefix + ".weight", {channels}), weights.get(prefix + ".bias", {channels}), groups, epsilon};
}

}  // namespace elman::model_weights
