#include "Backends/CpuParallel.h"
#include "kernel.h"
#include "Backends/CpuGemm.h"

namespace elman {
void launch_bmm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                    const uint32_t* c, const KernelParams& p) {
  if (try_gemm_cpu(out, a, b, c, p, CpuGemmOperation::Matmul)) return;
  parallel_for_cpu(p.output_count(), 128, [&](int64_t begin, int64_t end) {
    for (int i = static_cast<int>(begin); i < end; ++i) {
      int rows = p.output_shape(p.output_rank() - 2);
      int columns = p.output_shape(p.output_rank() - 1);
      int row = (i / columns) % rows;
      int column = i % columns;
      int batch = i / (rows * columns);
      int abase = matrix_batch_offset(batch, KernelInput::A, p);
      int bbase = matrix_batch_offset(batch, KernelInput::B, p);
      int shared_size =
          p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - 1);
      if (kernel_integer(p.output_dtype())) {
        uint32_t integer_value = 0u;
        for (int k = 0; k < shared_size; ++k) {
          integer_value += uint32_t(READ_INT_A(abase + row * shared_size + k)) *
                           uint32_t(READ_INT_B(bbase + k * columns + column));
        }
        WRITE_INT_OUTPUT(i, int(integer_value));
        continue;
      }
      float value = 0.0f;
      for (int k = 0; k < shared_size; ++k) {
        value += READ_A(abase + row * shared_size + k) *
                 READ_B(bbase + k * columns + column);
      }
      WRITE_OUTPUT(i, value);
    }
  });
}
}  // namespace elman
