#pragma once

#include <string>

#include "Runtime/RunOptions.h"

namespace elman {

int infer(const std::string& configuration, const RunOptions& options = {});

}  // namespace elman
