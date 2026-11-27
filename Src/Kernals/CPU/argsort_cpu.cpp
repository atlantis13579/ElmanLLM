#include "kernel.h"
#include "Core/AllocationPool.h"
#include "Kernals/sort_params.h"
#include "Kernals/integer_pointwise.h"
#include "Kernals/low_precision_pointwise_math.h"
#include <numeric>
#include <vector>

namespace elman {
void launch_argsort_cpu(uint32_t* out, const uint32_t* a, const uint32_t*,
                        const uint32_t*, const KernelParams& p) {
  if (!p.output_count()) return;
  int axis = p.attribute(kernal_attr::sorting::kAxis);
  int length = p.input_shape(KernelInput::A, axis), inner = p.input_stride(KernelInput::A, axis);
  bool descending = p.attribute(kernal_attr::sorting::kDescending) != 0;
  // The standard library still owns its merge scratch; reserve its upper bound.
  MemoryReservation reservation(Backend::CPU, checked_memory_mul(size_t(length), sizeof(int32_t)));
  BufferPtr workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(length), sizeof(int32_t)));
  int32_t* indices = static_cast<int32_t*>(workspace->raw_pointer());
  for (int group = 0; group < p.output_count() / length; ++group) {
    int base = group / inner * length * inner + group % inner;
    std::iota(indices, indices + length, 0);
    std::stable_sort(indices, indices + length, [&](int left, int right) {
      if (is_standard_integer(DType(p.input_dtype(KernelInput::A)))) {
        const int dtype = p.input_dtype(KernelInput::A);
        const uint64_t sign = integer_signed(dtype) ? uint64_t(1) << 63 : 0;
        const uint64_t x = integer_reduction_value(a, base + left * inner, dtype) ^ sign;
        const uint64_t y = integer_reduction_value(a, base + right * inner, dtype) ^ sign;
        return descending ? x > y : x < y;
      }
      const bool wide = p.input_dtype(KernelInput::A) == int(DType::Float64);
      const double x = wide ? reinterpret_cast<const double*>(a)[base + left * inner] : is_low_precision_float(DType(p.input_dtype(KernelInput::A))) ? low_precision_value(a, base + left * inner, DType(p.input_dtype(KernelInput::A))) : READ_A(base + left * inner);
      const double y = wide ? reinterpret_cast<const double*>(a)[base + right * inner] : is_low_precision_float(DType(p.input_dtype(KernelInput::A))) ? low_precision_value(a, base + right * inner, DType(p.input_dtype(KernelInput::A))) : READ_A(base + right * inner);
      if (std::isnan(x) || std::isnan(y))
        return descending ? std::isnan(x) && !std::isnan(y) : !std::isnan(x) && std::isnan(y);
      return descending ? x > y : x < y;
    });
    for (int j = 0; j < length; ++j) WRITE_INT_OUTPUT(base + j * inner, indices[j]);
  }
}
} // namespace elman
