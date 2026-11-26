#pragma once
#include "Core/Backend.h"
#include "Core/PackedInteger.h"

namespace elman {
inline bool packed_integer_cast_operation(const std::string& name, const KernelParams& p) {
    return name == "cast" && (is_packed_integer(DType(p.output_dtype())) ||
                             is_packed_integer(DType(p.input_dtype(KernelInput::A))));
}
void validate_packed_integer_cast(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
bool execute_packed_integer(const std::string& name, const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p);
void launch_packed_integer_cast_cpu(uint32_t* output, const uint32_t* input, const KernelParams& p);
void launch_packed_integer_cast_cuda(uint32_t* output, const uint32_t* input, const KernelParams& p);
} // namespace elman
