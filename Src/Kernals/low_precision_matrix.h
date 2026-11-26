#pragma once
#include "Core/Backend.h"

namespace elman {
// Matmul attributes select RHS transposition and folded FP32 batch arithmetic.
// The latter preserves the reference CPU GEMM/GEMV zero-sign behavior.
enum class LowPrecisionMatrix { None, Matmul, Bmm, Linear, PackedLinear, Addmm };

inline LowPrecisionMatrix low_precision_matrix_operation(const std::string& name, const KernelParams& p) {
    if (!is_low_precision_float(DType(p.output_dtype()))) return LowPrecisionMatrix::None;
    if (name == "matmul") return LowPrecisionMatrix::Matmul;
    if (name == "bmm") return LowPrecisionMatrix::Bmm;
    if (name == "linear") return LowPrecisionMatrix::Linear;
    if (name == "packed_linear") return LowPrecisionMatrix::PackedLinear;
    if (name == "addmm") return LowPrecisionMatrix::Addmm;
    return LowPrecisionMatrix::None;
}

void validate_low_precision_matrix(LowPrecisionMatrix operation, const BufferPtr& output,
    const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_low_precision_matrix_cpu(LowPrecisionMatrix operation, uint32_t* output,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_low_precision_matrix_cuda(LowPrecisionMatrix operation, uint32_t* output,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
