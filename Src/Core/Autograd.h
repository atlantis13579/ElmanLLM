#pragma once

#include "Core/Graph.h"

namespace elman {

namespace autograd {

void accumulate(const Tensor& tensor, const Tensor& gradient);

}  // namespace autograd
}  // namespace elman
