#pragma once

#include <string>

#include "Runtime/RunOptions.h"

namespace elman {

int train(const std::string& configuration, const RunOptions& options = {});
int validate_json(const std::string& path);

}  // namespace elman
