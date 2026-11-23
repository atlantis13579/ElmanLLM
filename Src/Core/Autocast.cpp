#include "Core/Autocast.h"

#include <stdexcept>

namespace elman {
namespace {
thread_local DType compute_dtype = DType::Float32;
}

AutocastScope::AutocastScope(DType dtype) : previous_(compute_dtype) {
  if (!is_compute_float(dtype)) throw std::invalid_argument("Autocast requires Float32, Float16 or BFloat16");
  compute_dtype = dtype;
}

AutocastScope::~AutocastScope() { compute_dtype = previous_; }

namespace autocast {
DType dtype() { return compute_dtype; }

bool should_cast(std::initializer_list<const Tensor*> operands, DType target) {
  if (compute_dtype == DType::Float32) return false;
  bool different = false;
  for (const Tensor* operand : operands) {
    if (!operand->defined()) continue;
    if (!is_compute_float(operand->dtype())) return false;
    different |= operand->dtype() != target;
  }
  return different;
}

Tensor cast_optional(const Tensor& tensor, DType target) { return tensor.defined() ? cast(tensor, target) : Tensor{}; }
}

}  // namespace elman
