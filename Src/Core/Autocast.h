#pragma once

#include <initializer_list>

#include "Core/Tensor.h"

namespace elman {

// Autocast selects an operation's compute dtype. Float32 disables the policy.
// The scope is thread local and applies to forward operations until destroyed.
class AutocastScope {
 public:
  explicit AutocastScope(DType dtype);
  ~AutocastScope();
  AutocastScope(const AutocastScope&) = delete;
  AutocastScope& operator=(const AutocastScope&) = delete;
 private:
  DType previous_;
};

namespace autocast {
DType dtype();
bool should_cast(std::initializer_list<const Tensor*> operands, DType target);
Tensor cast_optional(const Tensor& tensor, DType target);
}

}  // namespace elman
