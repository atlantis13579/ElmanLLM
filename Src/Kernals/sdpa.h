#pragma once
#include "Core/Backend.h"
namespace elman {
// Private stages of the standard SDPA kernal, not graph rewrite rules.
void validate_sdpa_stage(bool backward, const BufferPtr& output,
                         const std::vector<BufferPtr>& inputs, const KernelParams& params);
namespace sdpa_attr {
constexpr int scale = 0, causal = 1, seed = 2, threshold = 3, dropout_scale = 4;
}
}
