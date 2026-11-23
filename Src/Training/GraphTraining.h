#pragma once
#include <filesystem>
#include "Runtime/RunOptions.h"

namespace elman {
int train_graph(const std::filesystem::path& configuration, const RunOptions& options = {});
}
