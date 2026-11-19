#pragma once

#include <filesystem>

#include "Inference/InferenceSession.h"

namespace elman {

InferenceSession load_graph_session(const std::filesystem::path& path, Backend backend, int device);
InferenceSession make_graph_session(const Json& definition, NamedTensors weights, Backend backend, int device);

}  // namespace elman
