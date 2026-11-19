#pragma once

#include <filesystem>

#include "Inference/InferenceSession.h"

namespace elman {

struct VaeDecoder {
  InferenceSession session;
  double latent_scaling;
  int spatial_factor;
};

VaeDecoder load_diffusion_vae(const std::filesystem::path& directory, Backend backend, int device, DType dtype);

}  // namespace elman
