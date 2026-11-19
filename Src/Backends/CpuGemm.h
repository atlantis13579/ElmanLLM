#pragma once
#include "Core/Backend.h"

namespace elman {
enum class CpuGemmOperation { Matmul, Addmm, Linear };
std::string gemm_library_version_cpu();
int gemm_thread_count_cpu();
bool try_gemm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                  const uint32_t* c, const KernelParams& params, CpuGemmOperation operation);
} // namespace elman
