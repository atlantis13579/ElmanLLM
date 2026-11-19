#include "Operators/integer_matrix.h"

namespace elman {
void launch_integer_matrix_cpu(IntegerMatrix operation, uint32_t* output, const uint32_t* a,
                                const uint32_t* b, const uint32_t* c, const KernelParams& p) {
    for (int i = 0; i < p.output_count(); ++i)
        integer_write(output, i, p.output_dtype(), integer_matrix_value(operation, i, a, b, c, p));
}
} // namespace elman
