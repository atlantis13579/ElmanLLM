#include "Backends/CpuGemm.h"
#include "Backends/CpuParallel.h"
#include "Backends/AttentionGemm.h"
#include "Core/AllocationPool.h"
#include "Backends/CpuMath.h"
#include "Kernals/CPU/kernel.h"
#include "Kernals/matmul_params.h"

#ifdef ELMAN_HAS_MKL
#include <mkl.h>
#include <stdexcept>
#endif

namespace elman {
namespace {
#ifdef ELMAN_HAS_MKL
void initialize_blas() {
    static const bool ready = [] {
        if (mkl_set_interface_layer(MKL_INTERFACE_LP64) != MKL_INTERFACE_LP64)
            throw std::runtime_error("Cannot select the LP64 MKL interface");
#ifdef _WIN32
        // The Windows runtime package ships the Intel OpenMP threading layer.
        if (mkl_set_threading_layer(MKL_THREADING_INTEL) != MKL_THREADING_INTEL)
            throw std::runtime_error("Cannot select the Intel OpenMP MKL threading layer");
#endif
        return true;
    }();
    (void)ready;
}
struct LocalGemmThreads {
    bool enabled;
    int previous;
    explicit LocalGemmThreads(int threads) : enabled(threads > 0), previous(enabled ? mkl_set_num_threads_local(threads) : 0) {}
    ~LocalGemmThreads() { if (enabled) mkl_set_num_threads_local(previous); }
};
#endif
}

std::string gemm_library_version_cpu() {
#ifdef ELMAN_HAS_MKL
    initialize_blas();
    char version[256]{};
    mkl_get_version_string(version, sizeof(version));
    return version;
#else
    return "native";
#endif
}

int gemm_thread_count_cpu() {
#ifdef ELMAN_HAS_MKL
    initialize_blas();
    return mkl_get_max_threads();
#else
    return 0;
#endif
}

bool try_vml_math_cpu(float* output, const float* input, int count, CpuMathOperation operation) {
#ifdef ELMAN_HAS_MKL
    if (operation == CpuMathOperation::Silu) return false;
    if (count == 0) return true;
    initialize_blas();
    LocalGemmThreads threads(configured_cpu_thread_count());
    constexpr MKL_INT64 mode = VML_HA | VML_FTZDAZ_OFF | VML_ERRMODE_IGNORE;
    switch (operation) {
    case CpuMathOperation::Exp: vmsExp(count, input, output, mode); return true;
    case CpuMathOperation::Sin: vmsSin(count, input, output, mode); return true;
    case CpuMathOperation::Cos: vmsCos(count, input, output, mode); return true;
    default: break;
    }
#endif
    return false;
}

bool try_attention_gemm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b, const AttentionGemm& s) {
#ifndef ELMAN_HAS_MKL
    return false;
#else
    if (s.dtype != DType::Float32 || s.shared == 0 || int64_t(s.rows) * s.columns * s.shared < 400) return false;
    initialize_blas();
    // A head extracted from the reference batch retains its single-matrix
    // reduction order. Restore the caller's thread setting before returning.
    LocalGemmThreads threads(s.serial_cpu ? 1 : configured_cpu_thread_count());
    CBLAS_TRANSPOSE left_transpose = s.transpose_b ? CblasTrans : CblasNoTrans, right_transpose = CblasNoTrans;
    MKL_INT rows = s.rows, columns = s.columns, shared = s.shared, batches = s.batches;
    MKL_INT left_stride = s.transpose_b ? shared : columns, right_stride = shared, output_stride = columns;
    if (s.b_leading) left_stride = s.b_leading;
    float alpha = 1, beta = 0;
    if (batches == 1) {
        cblas_sgemm(CblasColMajor, left_transpose, right_transpose, columns, rows, shared, alpha,
                    reinterpret_cast<const float*>(b), left_stride, reinterpret_cast<const float*>(a), right_stride,
                    beta, reinterpret_cast<float*>(out), output_stride);
    } else {
        BufferPtr workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(batches), 3 * sizeof(void*)));
        const float** left = static_cast<const float**>(workspace->raw_pointer());
        const float** right = left + batches;
        float** outputs = reinterpret_cast<float**>(static_cast<char*>(workspace->raw_pointer()) + 2 * size_t(batches) * sizeof(void*));
        for (int i = 0; i < batches; ++i) {
            left[i] = reinterpret_cast<const float*>(b) + size_t(i / s.groups) * s.b_stride;
            right[i] = reinterpret_cast<const float*>(a) + size_t(i) * s.a_stride;
            outputs[i] = reinterpret_cast<float*>(out) + size_t(i) * s.output_stride;
        }
        cblas_sgemm_batch(CblasColMajor, &left_transpose, &right_transpose, &columns, &rows, &shared,
                          &alpha, left, &left_stride, right, &right_stride, &beta, outputs, &output_stride, 1, &batches);
    }
    return true;
#endif
}

bool try_gemm_cpu(uint32_t* out, const uint32_t* a, const uint32_t* b,
                  const uint32_t* c, const KernelParams& p, CpuGemmOperation operation) {
#ifndef ELMAN_HAS_MKL
    return false;
#else
    if (p.output_dtype() != int(DType::Float32) || p.input_dtype(KernelInput::A) != int(DType::Float32) ||
        p.input_dtype(KernelInput::B) != int(DType::Float32)) return false;
    if (p.output_count() == 0) return true;
    bool linear = operation == CpuGemmOperation::Linear;
    bool transposed = linear || (operation == CpuGemmOperation::Matmul && p.attribute(kernal_attr::matmul::kTransposeB) != 0);
    MKL_INT columns = p.output_shape(p.output_rank() - 1);
    MKL_INT rows = linear ? p.output_count() / columns : p.output_shape(p.output_rank() - 2);
    MKL_INT shared = p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - 1);
    MKL_INT batches = linear ? 1 : p.output_count() / (rows * columns);
    if (shared == 0) return false;
    // Match the reference's small batched-matrix path and preserve its scalar order.
    if (!linear && p.output_rank() > 2 && int64_t(rows) * columns * shared < 400) return false;
    float alpha = 1, beta = 0;
    if (operation == CpuGemmOperation::Addmm) {
        alpha = float_attribute(p, kernal_attr::addmm::kAlpha);
        beta = float_attribute(p, kernal_attr::addmm::kBeta);
    } else if (linear && p.input_rank(KernelInput::C) >= 0) beta = 1;
    if (alpha == 0) return false;
    if (beta != 0 && p.input_dtype(KernelInput::C) != int(DType::Float32)) return false;
    initialize_blas();
    float* destination = reinterpret_cast<float*>(out);
    LocalGemmThreads threads(configured_cpu_thread_count());
    bool separate_bias = linear && p.input_rank(KernelInput::C) == 0 && p.input_rank(KernelInput::A) != 2 && p.input_rank(KernelInput::A) != 3;
    if (beta != 0 && !separate_bias) {
        for (int i = 0; i < p.output_count(); ++i) {
            int index = linear ? (p.input_rank(KernelInput::C) == 0 ? 0 : i % columns) : broadcast_index(i, KernelInput::C, p);
            destination[i] = reinterpret_cast<const float*>(c)[index];
        }
    }
    if (separate_bias) beta = 0;
    // Row-major output is a column-major transpose; swap the two operands.
    CBLAS_TRANSPOSE left_transpose = transposed ? CblasTrans : CblasNoTrans, right_transpose = CblasNoTrans;
    MKL_INT left_stride = transposed ? shared : columns, right_stride = shared, output_stride = columns;
    if (batches == 1) {
        cblas_sgemm(CblasColMajor, left_transpose, right_transpose, columns, rows, shared, alpha,
                    reinterpret_cast<const float*>(b), left_stride, reinterpret_cast<const float*>(a), right_stride,
                    beta, destination, output_stride);
    } else {
        BufferPtr workspace = backend_workspace(Backend::CPU, checked_memory_mul(size_t(batches), 3 * sizeof(void*)));
        const float** left = static_cast<const float**>(workspace->raw_pointer());
        const float** right = left + batches;
        float** outputs = reinterpret_cast<float**>(static_cast<char*>(workspace->raw_pointer()) + 2 * size_t(batches) * sizeof(void*));
        for (int i = 0; i < batches; ++i) {
            left[i] = reinterpret_cast<const float*>(b) + matrix_batch_offset(i, KernelInput::B, p);
            right[i] = reinterpret_cast<const float*>(a) + matrix_batch_offset(i, KernelInput::A, p);
            outputs[i] = destination + i * rows * columns;
        }
        cblas_sgemm_batch(CblasColMajor, &left_transpose, &right_transpose, &columns, &rows, &shared,
                          &alpha, left, &left_stride, right, &right_stride, &beta,
                          outputs, &output_stride, 1, &batches);
    }
    if (separate_bias) {
        float bias = reinterpret_cast<const float*>(c)[0];
        for (int i = 0; i < p.output_count(); ++i) destination[i] += bias;
    }
    return true;
#endif
}
} // namespace elman
