#include "kernel.h"
#include "Kernals/sort_params.h"

namespace elman {
void launch_sort_backward_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                              const uint32_t*, const KernelParams& p) {
  int axis = p.attribute(operator_attr::sorting::kAxis);
  int length = p.output_shape(axis), inner = p.input_stride(KernelInput::A, axis);
  for (int i = 0; i < p.output_count(); ++i) {
    int target = i / (length * inner) * length * inner + i % inner + READ_INT_B(i) * inner;
    // Sorting produces a permutation, so each destination is written once.
    WRITE_OUTPUT(target, READ_A(i));
  }
}
} // namespace elman
