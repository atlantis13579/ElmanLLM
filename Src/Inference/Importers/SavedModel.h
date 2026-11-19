#pragma once

#include "Inference/InferenceSession.h"

namespace elman {

InferenceSession load_saved_model_session(const std::filesystem::path& path, Backend backend, int device,
                                          const std::string& signature = "serving_default");

}  // namespace elman
