#pragma once

#include <filesystem>

#include "Inference/InferenceSession.h"

namespace elman {

InferenceSession load_diffusion_unet(const std::filesystem::path& directory, Backend backend,
                                      int device, DType dtype);

}  // namespace elman
