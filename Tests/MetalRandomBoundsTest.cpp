#include "Kernals/Metal/random_bounds.h"

#include <iostream>
#include <stdexcept>

int main() {
  try {
    using elman::DType;
    using elman::metal::uniform_upper_bound;
    for (DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
      const double step = dtype == DType::Float32 ? 0x1p-23 : dtype == DType::Float16 ? 0x1p-10 : 0x1p-7;
      for (double base : {1.0, -1.0}) {
        // high rounds down to base. The old unconditional decrement fell below low.
        if (uniform_upper_bound(dtype, base, base + step / 8) != base)
          throw std::runtime_error("Rounded-down upper endpoint lost its valid sample");
        // Both rounded endpoints coincide, including rounding upward.
        if (uniform_upper_bound(dtype, base - step / 8, base) != base)
          throw std::runtime_error("Collapsed interval fell below its rounded lower endpoint");
      }
      const float cap = uniform_upper_bound(dtype, -1, 1);
      if (!(cap >= -1 && cap < 1)) throw std::runtime_error("Upper endpoint is not half-open");
      if (!(uniform_upper_bound(dtype, -1, 0) < 0))
        throw std::runtime_error("Zero endpoint did not step toward negative infinity");
      if (uniform_upper_bound(dtype, 2, 2) != 2)
        throw std::runtime_error("Constant uniform interval changed");
    }
    std::cout << "Metal uniform bounds passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
