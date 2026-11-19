#pragma once

#include "Core/Backend.h"

namespace elman {
// Returns the first maximum, or -1 if any Float32 logit is nonfinite.
int32_t greedy_id_cuda(const BufferPtr& logits);
}
