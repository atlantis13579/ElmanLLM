#pragma once
#include "Core/Backend.h"

namespace elman {
void validate_floating_dropout(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
} // namespace elman
