#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include "Core/Tensor.h"

namespace elman {

size_t shape_element_count(const Shape& shape);

namespace detail {
BufferPtr allocate_dispatch_output(const std::string& op, Backend backend, size_t count, DType dtype,
                                   const std::vector<Tensor>& inputs);
}

Tensor dispatch_tensor(const std::string& operation, const Shape& shape,
                       const std::vector<Tensor>& inputs,
                       const std::vector<int>& attributes = {},
                       DType dtype = DType::Float32,
                       Backend backend = Backend::CPU);

// Internal mutable dispatch. Callers own graph/version semantics for output.
void dispatch_tensor_into(const std::string& operation, const Tensor& output,
                          const std::vector<Tensor>& inputs,
                          const std::vector<int>& attributes = {});

}  // namespace elman
