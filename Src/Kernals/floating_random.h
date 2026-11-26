#pragma once
#include "Core/Backend.h"

namespace elman {
void validate_floating_random(bool normal, const BufferPtr& output,
    const std::vector<BufferPtr>& inputs, const KernelParams& params);
} // namespace elman
