#pragma once

#include <filesystem>

#include "Inference/InferenceSession.h"

namespace elman {

InferenceSession load_clip_text_encoder(const std::filesystem::path& directory, Backend backend,
                                         int device, DType dtype);

}  // namespace elman
