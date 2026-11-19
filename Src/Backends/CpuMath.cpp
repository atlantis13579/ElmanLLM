#include "Backends/CpuMath.h"
#include "Backends/CpuGemm.h"
#include "Backends/CpuParallel.h"
#include <cstdlib>
#include <limits>
#if defined(ELMAN_HAS_SLEEF) && defined(_MSC_VER)
#include <intrin.h>
#endif

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
#ifdef ELMAN_HAS_SLEEF
    static const bool available = [] {
#ifdef _MSC_VER
        int registers[4];
        __cpuid(registers, 1);
        constexpr int flags = (1 << 12) | (1 << 27) | (1 << 28);
        if ((registers[2] & flags) != flags || (_xgetbv(0) & 6) != 6) return false;
        __cpuidex(registers, 7, 0);
        return (registers[1] & (1 << 5)) != 0;
#else
        return bool(__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"));
#endif
    }();
    return available;
#else
    return false;
#endif
}
#ifdef ELMAN_HAS_SLEEF
void unary_math_avx2(float*, const float*, int, CpuMathOperation);
void softmax_math_avx2(float*, const float*, int, int, bool);
std::string vector_math_version_avx2();
#endif
std::string cpu_vector_math_version() {
#ifdef ELMAN_HAS_SLEEF
    if (cpu_vector_math_available()) return vector_math_version_avx2();
#endif
    return "native";
}
bool try_unary_math_cpu(float* output, const float* input, int count, CpuMathOperation operation) {
    // Standalone transcendental operators use MKL VML when available. Composite
    // activation/softmax kernels use SLEEF, matching their independent formulas.
    if (try_vml_math_cpu(output, input, count, operation)) return true;
#ifdef ELMAN_HAS_SLEEF
    if (cpu_vector_math_available()) { unary_math_avx2(output, input, count, operation); return true; }
#endif
    return false;
}
bool try_softmax_math_cpu(float* output, const float* input, int rows, int columns, bool narrow_input) {
#ifdef ELMAN_HAS_SLEEF
    if (cpu_vector_math_available()) { softmax_math_avx2(output, input, rows, columns, narrow_input); return true; }
#endif
    return false;
}
} // namespace elman
