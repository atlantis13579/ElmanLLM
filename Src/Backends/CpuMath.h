#pragma once
#include <string>

namespace elman {
enum class CpuMathOperation { Exp, Sin, Cos, Silu };
bool cpu_vector_math_available();
int cpu_math_thread_count();
std::string cpu_vector_math_version();
bool try_vml_math_cpu(float* output, const float* input, int count, CpuMathOperation operation);
bool try_unary_math_cpu(float* output, const float* input, int count, CpuMathOperation operation);
bool try_softmax_math_cpu(float* output, const float* input, int rows, int columns, bool narrow_input = false);
} // namespace elman
