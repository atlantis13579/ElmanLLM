#pragma once

#include "Inference/InferenceSession.h"

namespace elman {

InferenceSession load_tflite_session(const std::filesystem::path& path, Backend backend, int device);

}  // namespace elman
