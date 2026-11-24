#include "Backends/CpuMath.h"
#include "Backends/CpuGemm.h"
#include "Backends/CpuParallel.h"
#include <cstdlib>
#include <limits>

namespace elman {
int cpu_math_thread_count() {
    const int configured = configured_cpu_thread_count();
    if (configured) return configured;
    const int blas_threads = gemm_thread_count_cpu();
    if (blas_threads > 0) return blas_threads;
    // Keep deterministic reduction partitions consistent with the configured
    // CPU worker count even when this build has no external BLAS runtime.
    for (const char* name : {"MKL_NUM_THREADS", "OMP_NUM_THREADS"}) {
        const char* value = std::getenv(name);
        if (!value || !*value) continue;
        char* end = nullptr;
        const long count = std::strtol(value, &end, 10);
        if (end != value && !*end && count > 0 && count <= std::numeric_limits<int>::max()) return int(count);
    }
    return 1;
}

bool cpu_vector_math_available() {
    return false;
}
std::string cpu_vector_math_version() {
    return "native";
}
bool try_unary_math_cpu(float* output, const float* input, int count, CpuMathOperation operation) {
    return try_vml_math_cpu(output, input, count, operation);
}
bool try_softmax_math_cpu(float*, const float*, int, int, bool) {
    return false;
}
} // namespace elman
