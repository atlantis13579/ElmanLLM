#pragma once

#include <stdexcept>
#include <vector>

#include "IO/Json.h"

namespace elman::diffusion {

inline std::vector<int> block_channels(const Json& config) {
  std::vector<int> channels;
  for (const Json& value : config.at("block_out_channels").array()) {
    const int width = value.integer();
    if (width <= 0 || width > 65536) throw std::invalid_argument("Invalid diffusion block width");
    channels.push_back(width);
  }
  if (channels.empty() || channels.size() > 8) throw std::invalid_argument("Unsupported diffusion block count");
  return channels;
}

}  // namespace elman::diffusion
