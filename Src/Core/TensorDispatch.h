#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Core/Tensor.h"

namespace elman {

size_t shape_element_count(const Shape& shape);

Tensor dispatch_tensor(const std::string& operation, const Shape& shape,
                       const std::vector<Tensor>& inputs,
                       const std::vector<int>& attributes = {},
                       DType dtype = DType::Float32,
                       Backend backend = Backend::CPU);

}  // namespace elman
