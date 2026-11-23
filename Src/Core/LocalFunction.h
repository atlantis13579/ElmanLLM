#pragma once

namespace elman {

// Preserve a local lambda's concrete type without type erasure.
template<class Function>
struct LocalFunction : Function {
  using Function::operator();
};

template<class Function>
LocalFunction(Function) -> LocalFunction<Function>;

} // namespace elman
