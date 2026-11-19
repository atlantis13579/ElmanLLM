#pragma once
#include "Core/Tensor.h"

namespace elman {
DType complex_promoted_dtype(const Tensor& a, const Tensor& b);
Tensor complex_binary(const std::string& operation, const Tensor& a, const Tensor& b);
Tensor complex_unary(const std::string& operation, const Tensor& x);
Tensor complex_matrix(const Tensor& a, const Tensor& b);
Tensor complex_reduce(const std::string& operation, const Tensor& x, int dimension, bool keepdim, bool all);
} // namespace elman
