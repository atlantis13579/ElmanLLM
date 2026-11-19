#pragma once
#include "Core/Backend.h"

namespace elman {
enum class GemmOperation { Matmul, Addmm, Linear };
struct GemmLibraryVersions { int blas, blas_lt; };
GemmLibraryVersions gemm_library_versions_cuda();
bool try_biased_float64_gemm_cuda(double* out, const double* a, const double* b, const double* bias,
                                 int rows, int columns, int shared, bool linear, double alpha);
bool try_gemm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                   const uint32_t* c, const KernelParams& params, GemmOperation operation);
bool try_conv2d_gemm_cuda(void* output, const void* columns, const void* weights, DType dtype,
                         int positions, int channels, int shared, int input_pitch, int output_pitch,
                         bool output_float32, int groups);
void convolution_filter_partials_cuda(float* partials, const float* packed, int patches, int outputs,
                                      int tile, int tiles);
} // namespace elman
