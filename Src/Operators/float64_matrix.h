#pragma once
#include "Core/Backend.h"

namespace elman {
enum class Float64Matrix { None, Matmul, Bmm, Linear, PackedLinear, Addmm };

inline Float64Matrix float64_matrix_operation(const std::string& name, const KernelParams& p) {
    if (DType(p.output_dtype()) != DType::Float64) return Float64Matrix::None;
    if (name == "matmul") return Float64Matrix::Matmul;
    if (name == "bmm") return Float64Matrix::Bmm;
    if (name == "linear") return Float64Matrix::Linear;
    if (name == "packed_linear") return Float64Matrix::PackedLinear;
    if (name == "addmm") return Float64Matrix::Addmm;
    return Float64Matrix::None;
}

void validate_float64_matrix(Float64Matrix operation, const BufferPtr& output,
    const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_float64_matrix_cpu(Float64Matrix operation, uint32_t* output,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
void launch_float64_matrix_cuda(Float64Matrix operation, uint32_t* output,
    const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p);
} // namespace elman
