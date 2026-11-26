#include "Backends/CudaGemm.h"
#include "Backends/CudaConvolution.h"
#include "Backends/AttentionGemm.h"
#include "Core/Memory.h"
#include "Backends/CudaExecution.h"
#include "Kernals/CUDA/kernel.cuh"
#include "Kernals/matmul_params.h"
#include "Kernals/conv2d_params.h"
#include "Kernals/low_precision_attention.h"

#include <cublas_v2.h>
#include <cublasLt.h>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace elman {
namespace {
void check(cublasStatus_t status, const char* operation) {
  if (status != CUBLAS_STATUS_SUCCESS)
    throw std::runtime_error(std::string(operation) + ": " + cublasGetStatusString(status));
}
bool& tf32_setting() {
  static thread_local bool enabled = false;
  return enabled;
}
bool& conv_tf32_setting() {
  static thread_local bool enabled = false;
  return enabled;
}
bool allow_tf32() { return tf32_setting(); }
cublasMath_t selected_math_mode() {
  return static_cast<cublasMath_t>(
      (allow_tf32() ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH) |
      CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
}
cublasComputeType_t compute32(cudaDataType_t storage) {
  return allow_tf32() && storage == CUDA_R_32F ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
}
cublasComputeType_t compute_conv32(cudaDataType_t storage) {
  if (storage != CUDA_R_32F) return CUBLAS_COMPUTE_32F;
  if (conv_tf32_setting()) return CUBLAS_COMPUTE_32F_FAST_TF32;
  return allow_tf32() ? CUBLAS_COMPUTE_32F_PEDANTIC : CUBLAS_COMPUTE_32F;
}
struct LtDescriptors {
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t left = nullptr, right = nullptr, output = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulHeuristicResult_t algorithm{};
  bool supported = false;
  ~LtDescriptors() {
    if (preference) cublasLtMatmulPreferenceDestroy(preference);
    if (output) cublasLtMatrixLayoutDestroy(output);
    if (right) cublasLtMatrixLayoutDestroy(right);
    if (left) cublasLtMatrixLayoutDestroy(left);
    if (operation) cublasLtMatmulDescDestroy(operation);
  }
};
struct BlasContext {
  std::map<std::array<int,10>, std::unique_ptr<LtDescriptors>> plans;
  MemoryReservation reservation;
  cublasHandle_t handle = nullptr;
  cublasLtHandle_t lt = nullptr;
  void* workspace = nullptr;
  int device = 0;
  bool tf32 = allow_tf32();
  static constexpr size_t workspace_bytes = 1024 * 1024;
  BlasContext() {
    check_cuda_execution(cudaGetDevice(&device), "GEMM device");
    check(cublasCreate(&handle), "cublasCreate");
    try {
      check(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST), "cublasSetPointerMode");
      check(cublasSetMathMode(handle, selected_math_mode()), "cublasSetMathMode");
      check(cublasSetAtomicsMode(handle, CUBLAS_ATOMICS_NOT_ALLOWED), "cublasSetAtomicsMode");
      check(cublasSetStream(handle, current_cuda_stream()), "cublasSetStream");
      check(cublasLtCreate(&lt), "cublasLtCreate");
      reservation = MemoryReservation(Backend::CUDA, workspace_bytes);
      if (cudaMalloc(&workspace, workspace_bytes) != cudaSuccess) {
        cudaGetLastError();
        throw std::runtime_error("Cannot allocate GEMM workspace");
      }
    } catch (...) { if (lt) cublasLtDestroy(lt); cublasDestroy(handle); throw; }
  }
  ~BlasContext() {
    int previous = -1;
    cudaGetDevice(&previous);
    if (previous != device) cudaSetDevice(device);
    plans.clear();
    if (workspace) cudaFree(workspace);
    if (lt) cublasLtDestroy(lt);
    if (handle) cublasDestroy(handle);
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
};
BlasContext& context() {
  std::shared_ptr<void>& context = current_cuda_blas();
  if (!context) context = std::make_shared<BlasContext>();
  BlasContext& blas = *static_cast<BlasContext*>(context.get());
  if (blas.tf32 != allow_tf32()) {
    blas.plans.clear();
    check(cublasSetMathMode(blas.handle, selected_math_mode()), "cublasSetMathMode");
    blas.tf32 = allow_tf32();
  }
  return blas;
}
cudaDataType_t gemm_storage(DType dtype) {
  return dtype == DType::Float32 ? CUDA_R_32F : dtype == DType::Float16 ? CUDA_R_16F : CUDA_R_16BF;
}
uint32_t lt_alignment(const void* pointer) {
  const uintptr_t address = reinterpret_cast<uintptr_t>(pointer);
  uint32_t alignment = 256;
  while (address % alignment) alignment /= 2;
  return alignment;
}
void prepare_lt_plan(LtDescriptors& d, BlasContext& blas, cudaDataType_t dtype, bool wide, bool linear,
                     int rows, int columns, int shared, const void* bias,
                     uint32_t align_left, uint32_t align_right, uint32_t align_output) {
  check(cublasLtMatmulDescCreate(&d.operation, wide ? CUBLAS_COMPUTE_64F : compute32(dtype),
                                wide ? CUDA_R_64F : CUDA_R_32F), "cublasLtMatmulDescCreate");
  cublasOperation_t transpose = linear ? CUBLAS_OP_T : CUBLAS_OP_N;
  check(cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose, sizeof(transpose)), "Lt transpose");
  cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
  check(cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue, sizeof(epilogue)), "Lt bias epilogue");
  check(cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)), "Lt bias pointer");
  check(cublasLtMatrixLayoutCreate(&d.left, dtype, linear ? shared : columns, linear ? columns : shared,
                                   linear ? shared : columns), "Lt left layout");
  check(cublasLtMatrixLayoutCreate(&d.right, dtype, shared, rows, shared), "Lt right layout");
  check(cublasLtMatrixLayoutCreate(&d.output, dtype, columns, rows, columns), "Lt output layout");
  check(cublasLtMatmulPreferenceCreate(&d.preference), "Lt preference");
  check(cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_left, sizeof(align_left)), "Lt left alignment");
  check(cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_right, sizeof(align_right)), "Lt right alignment");
  check(cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_output, sizeof(align_output)), "Lt addend alignment");
  check(cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, &align_output, sizeof(align_output)), "Lt output alignment");
  size_t workspace_bytes = BlasContext::workspace_bytes;
  check(cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_bytes, sizeof(workspace_bytes)), "Lt workspace preference");
  uint32_t reduction = CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE | CUBLASLT_REDUCTION_SCHEME_NONE;
  check(cublasLtMatmulPreferenceSetAttribute(d.preference, CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
                                             &reduction, sizeof(reduction)), "Lt reduction precision");
  int count = 0;
  cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(blas.lt, d.operation, d.left, d.right, d.output, d.output,
                                             d.preference, 1, &d.algorithm, &count);
  if (status != CUBLAS_STATUS_NOT_SUPPORTED) {
    check(status, "cublasLtMatmulAlgoGetHeuristic");
    d.supported = count > 0;
  }
}
template<class Scalar>
bool biased_gemm(BlasContext& blas, void* out, const void* a, const void* b, const void* bias,
                 int rows, int columns, int shared, bool linear, cudaDataType_t dtype, Scalar alpha) {
  constexpr bool wide = std::is_same_v<Scalar,double>;
  const uint32_t left = lt_alignment(b), right = lt_alignment(a), dest = lt_alignment(out);
  const std::array<int,10> key{rows,columns,shared,int(dtype),int(wide),int(linear),
                               int(left),int(right),int(dest),int(lt_alignment(bias))};
  auto it = blas.plans.find(key);
  if (it == blas.plans.end()) {
    std::unique_ptr<LtDescriptors> plan = std::make_unique<LtDescriptors>();
    prepare_lt_plan(*plan, blas, dtype, wide, linear, rows, columns, shared, bias, left, right, dest);
    if (blas.plans.size() >= 128) blas.plans.clear();
    it = blas.plans.emplace(key, std::move(plan)).first;
  }
  LtDescriptors& d = *it->second;
  if (!d.supported) return false;
  check(cublasLtMatmulDescSetAttribute(d.operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)), "Lt bias pointer");
  Scalar beta = 0;
  cublasStatus_t status = cublasLtMatmul(blas.lt, d.operation, &alpha, b, d.left, a, d.right, &beta, out, d.output,
                         out, d.output, &d.algorithm.algo, blas.workspace, BlasContext::workspace_bytes, current_cuda_stream());
  if (status == CUBLAS_STATUS_NOT_SUPPORTED) return false;
  check(status, "cublasLtMatmul");
  return true;
}
float attribute(const KernelParams& p, int index) {
  int bits = p.attribute(index);
  float value;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}
int batch_offset(int flat, KernelInput input, const KernelParams& p) {
  int result = 0;
  int delta = p.output_rank() - p.input_rank(input);
  for (int axis = p.output_rank() - 3; axis >= 0; --axis) {
    int coordinate = flat % p.output_shape(axis);
    flat /= p.output_shape(axis);
    int input_axis = axis - delta;
    if (input_axis >= 0 && p.input_shape(input, input_axis) != 1)
      result += coordinate * p.input_stride(input, input_axis);
  }
  return result;
}
__global__ void seed_addend(uint32_t* out, const uint32_t* c, KernelParams p) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < unsigned(p.output_count())) {
    const int source = broadcast_index(int(i), KernelInput::C, p);
    COPY_TYPED_ELEMENT(out, c, int(i), source, p.output_dtype());
  }
}
__global__ void attention_pointers(void** pointers, uint32_t* out, const uint32_t* a, const uint32_t* b,
                                    AttentionGemm s, int bytes) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= s.batches) return;
  pointers[i] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(b) + size_t(i / s.groups) * s.b_stride * bytes);
  pointers[size_t(s.batches) + i] = const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(a) + size_t(i) * s.a_stride * bytes);
  pointers[2 * size_t(s.batches) + i] = reinterpret_cast<uint8_t*>(out) + size_t(i) * s.output_stride * bytes;
}
__global__ void attention_empty_gemm(uint32_t* out, AttentionGemm s) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x), matrix = s.rows * s.columns;
  if (i < s.batches * matrix)
    elman_cuda_store(out, (i / matrix) * s.output_stride + i % matrix, 0, int(s.dtype));
}
}

void set_cuda_tf32_math(bool matmul, bool convolution) {
  conv_tf32_setting() = convolution;
  if (tf32_setting() == matmul) return;
  tf32_setting() = matmul;
  std::shared_ptr<void>& active = current_cuda_blas();
  if (active) {
    BlasContext& blas = *static_cast<BlasContext*>(active.get());
    blas.plans.clear();
    check(cublasSetMathMode(blas.handle, selected_math_mode()), "cublasSetMathMode");
    blas.tf32 = matmul;
  }
}
bool cuda_conv_tf32_enabled() { return conv_tf32_setting(); }

GemmLibraryVersions gemm_library_versions_cuda() {
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  int version = 0;
  check(cublasGetVersion(context().handle, &version), "cublasGetVersion");
  return {version, static_cast<int>(cublasLtGetVersion())};
}

void attention_gemm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                         void* pointers, const AttentionGemm& s) {
  if (!s.rows || !s.columns || !s.batches) return;
  if (!s.shared) {
    unsigned count = unsigned(s.batches) * unsigned(s.rows) * unsigned(s.columns);
    attention_empty_gemm<<<(count + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, s);
    return;
  }
  BlasContext& blas = context();
  cublasOperation_t transpose = s.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cudaDataType_t storage = gemm_storage(s.dtype);
  float alpha = 1, beta = 0;
  int leading_b = s.b_leading ? s.b_leading : s.transpose_b ? s.shared : s.columns;
  if (s.batches == 1) {
    check(cublasGemmEx(blas.handle, transpose, CUBLAS_OP_N, s.columns, s.rows, s.shared,
                      &alpha, b, storage, leading_b, a, storage, s.shared, &beta, out, storage, s.columns,
                      compute32(storage), CUBLAS_GEMM_DEFAULT), "attention cublasGemmEx");
  } else if (s.groups == 1) {
    check(cublasGemmStridedBatchedEx(blas.handle, transpose, CUBLAS_OP_N, s.columns, s.rows, s.shared,
                &alpha, b, storage, leading_b, s.b_stride, a, storage, s.shared, s.a_stride,
                &beta, out, storage, s.columns, s.output_stride, s.batches, compute32(storage),
                CUBLAS_GEMM_DEFAULT), "attention cublasGemmStridedBatchedEx");
  } else {
    void** arrays = static_cast<void**>(pointers);
    attention_pointers<<<(unsigned(s.batches) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(
        arrays, out, a, b, s, int(dtype_storage_bytes(s.dtype, 1)));
    check(cublasGemmBatchedEx(blas.handle, transpose, CUBLAS_OP_N, s.columns, s.rows, s.shared,
                &alpha, reinterpret_cast<const void* const*>(arrays), storage, leading_b,
                reinterpret_cast<const void* const*>(arrays + s.batches), storage, s.shared, &beta,
                arrays + 2 * size_t(s.batches), storage, s.columns, s.batches, compute32(storage),
                CUBLAS_GEMM_DEFAULT), "attention cublasGemmBatchedEx");
  }
}

void attention_gemm_cuda_pointers(void* pointers, const AttentionGemm& s) {
  BlasContext& blas = context();
  const cublasOperation_t transpose = s.transpose_b ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cudaDataType_t storage = gemm_storage(s.dtype);
  const int leading = s.b_leading ? s.b_leading : s.transpose_b ? s.shared : s.columns;
  void** arrays = static_cast<void**>(pointers);
  float alpha = 1, beta = 0;
  check(cublasGemmBatchedEx(blas.handle, transpose, CUBLAS_OP_N, s.columns, s.rows, s.shared,
      &alpha, reinterpret_cast<const void* const*>(arrays), storage, leading,
      reinterpret_cast<const void* const*>(arrays + s.batches), storage, s.shared, &beta,
      arrays + 2 * size_t(s.batches), storage, s.columns, s.batches, compute32(storage), CUBLAS_GEMM_DEFAULT), "paged cublasGemmBatchedEx");
}

void launch_low_attention_matrix_cuda(uint32_t* output, const uint32_t* a, const uint32_t* b, const KernelParams& p) {
  if (!p.output_count()) return;
  const int rows = p.output_shape(2), columns = p.output_shape(3), shared = p.input_shape(KernelInput::A, p.attribute(0) ? 2 : 3);
  if (!shared) {
    check_cuda_execution(cudaMemsetAsync(output, 0, size_t(p.output_count()) * sizeof(float), current_cuda_stream()), "attention empty matrix");
    return;
  }
  BlasContext& blas = context();
  const cublasOperation_t left = p.attribute(1) ? CUBLAS_OP_T : CUBLAS_OP_N;
  const cublasOperation_t right = p.attribute(0) ? CUBLAS_OP_T : CUBLAS_OP_N;
  const int leading_left = p.attribute(1) ? shared : columns, leading_right = p.attribute(0) ? rows : shared;
  const int batches = p.output_shape(0) * p.output_shape(1);
  float alpha = 1, beta = 0;
  if (batches == 1)
    check(cublasGemmEx(blas.handle, left, right, columns, rows, shared, &alpha,
      b, CUDA_R_32F, leading_left, a, CUDA_R_32F, leading_right, &beta, output, CUDA_R_32F, columns,
      compute32(CUDA_R_32F), CUBLAS_GEMM_DEFAULT), "attention FP32 cublasGemmEx");
  else
    check(cublasGemmStridedBatchedEx(blas.handle, left, right, columns, rows, shared, &alpha,
      b, CUDA_R_32F, leading_left, int64_t(columns) * shared, a, CUDA_R_32F, leading_right, int64_t(rows) * shared,
      &beta, output, CUDA_R_32F, columns, int64_t(rows) * columns, batches,
      compute32(CUDA_R_32F), CUBLAS_GEMM_DEFAULT), "attention FP32 cublasGemmStridedBatchedEx");
}

bool try_biased_float64_gemm_cuda(double* out, const double* a, const double* b, const double* bias,
                                 int rows, int columns, int shared, bool linear, double alpha) {
  return biased_gemm(context(), out, a, b, bias, rows, columns, shared, linear, CUDA_R_64F, alpha);
}

bool try_conv2d_gemm_cuda(void* output, const void* columns, const void* weights, DType dtype,
                         int positions, int channels, int shared, int input_pitch, int output_pitch,
                         bool output_float32, int groups) {
  const cudaDataType_t storage = gemm_storage(dtype);
  const cudaDataType_t destination = output_float32 ? CUDA_R_32F : storage;
  const float alpha = 1, beta = 0;
  // Each column of the output matrix is one channel plane in NCHW storage.
  cublasHandle_t handle = context().handle;
  cublasStatus_t status;
  if (groups == 1) {
    status = cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, positions, channels, shared,
                         &alpha, columns, storage, input_pitch, weights, storage, shared, &beta,
                         output, destination, output_pitch, compute_conv32(storage), CUBLAS_GEMM_DEFAULT);
  } else {
    status = cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, positions, channels, shared,
        &alpha, columns, storage, input_pitch, int64_t(shared) * input_pitch,
        weights, storage, shared, int64_t(shared) * channels, &beta,
        output, destination, output_pitch, int64_t(channels) * output_pitch,
        groups, compute_conv32(storage), CUBLAS_GEMM_DEFAULT);
  }
  if (status == CUBLAS_STATUS_NOT_SUPPORTED || status == CUBLAS_STATUS_ARCH_MISMATCH) return false;
  check(status, "conv2d cublasGemmEx");
  return true;
}

void convolution_filter_partials_cuda(float* partials, const float* packed, int patches, int outputs,
                                      int tile, int tiles) {
  const int64_t stride = int64_t(patches + outputs) * tile;
  const float one = 1.f, zero = 0.f;
  check(cublasGemmStridedBatchedEx(context().handle, CUBLAS_OP_T, CUBLAS_OP_N, patches, outputs, tile,
        &one, packed, CUDA_R_32F, tile, stride,
        packed + size_t(patches) * tile, CUDA_R_32F, tile, stride,
        &zero, partials, CUDA_R_32F, patches, int64_t(patches) * outputs, tiles,
        compute_conv32(CUDA_R_32F), CUBLAS_GEMM_DEFAULT), "Convolution filter gradient tiles");
}

bool try_pointwise_conv_gradient_cuda(CudaConvolution operation, uint32_t* out,
                                       const uint32_t* a, const uint32_t* b, const KernelParams& p) {
  using namespace operator_attr::conv2d;
  constexpr int fp32 = int(DType::Float32);
  if (p.output_rank() != 4 || p.input_rank(KernelInput::A) != 4 || p.input_rank(KernelInput::B) != 4 ||
      p.output_dtype() != fp32 || p.input_dtype(KernelInput::A) != fp32 || p.input_dtype(KernelInput::B) != fp32 ||
      p.attribute(kGroups) != 1 || p.attribute(kStrideHeight) != 1 || p.attribute(kStrideWidth) != 1 ||
      p.attribute(kPaddingHeight) != 0 || p.attribute(kPaddingWidth) != 0) return false;
  const bool dx = operation == CudaConvolution::InputGradient;
  const int kernel_h = dx ? p.input_shape(KernelInput::B, kHeightAxis) : p.output_shape(kHeightAxis);
  const int kernel_w = dx ? p.input_shape(KernelInput::B, kWidthAxis) : p.output_shape(kWidthAxis);
  const int batches = p.input_shape(KernelInput::A, kBatchAxis);
  const int positions = p.input_shape(KernelInput::A, kHeightAxis) * p.input_shape(KernelInput::A, kWidthAxis);
  const int inputs = dx ? p.output_shape(kChannelAxis) : p.input_shape(KernelInput::A, kChannelAxis);
  const int outputs = dx ? p.input_shape(KernelInput::A, kChannelAxis) : p.output_shape(kOutputChannelAxis);
  if (kernel_h != 1 || kernel_w != 1 || !batches || !positions || int64_t(positions) * inputs * outputs < (1 << 15))
    return false;
  cublasHandle_t handle = context().handle;
  const float* left = reinterpret_cast<const float*>(a);
  const float* right = reinterpret_cast<const float*>(b);
  float* destination = reinterpret_cast<float*>(out);
  const float one = 1, zero = 0;
  if (dx) {
    check(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T, positions, inputs, outputs,
        &one, left, CUDA_R_32F, positions, int64_t(outputs) * positions,
        right, CUDA_R_32F, inputs, 0,
        &zero, destination, CUDA_R_32F, positions, int64_t(inputs) * positions, batches,
        compute_conv32(CUDA_R_32F), CUBLAS_GEMM_DEFAULT), "Pointwise convolution input gradient");
  } else {
    for (int batch = 0; batch < batches; ++batch) {
      const float* x = left + size_t(batch) * inputs * positions;
      const float* dy = right + size_t(batch) * outputs * positions;
      check(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, inputs, outputs, positions, &one,
          x, CUDA_R_32F, positions, dy, CUDA_R_32F, positions,
          batch ? &one : &zero, destination, CUDA_R_32F, inputs,
          compute_conv32(CUDA_R_32F), CUBLAS_GEMM_DEFAULT), "Pointwise convolution weight gradient");
    }
  }
  return true;
}

bool try_gemm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                   const uint32_t* c, const KernelParams& p, GemmOperation operation) {
  const DType dtype = static_cast<DType>(p.output_dtype());
  if (!is_compute_float(dtype) || p.input_dtype(KernelInput::A) != p.output_dtype() ||
      p.input_dtype(KernelInput::B) != p.output_dtype()) return false;
  if (p.output_count() == 0) return true;
  const bool linear = operation == GemmOperation::Linear;
  const bool transposed = linear || (operation == GemmOperation::Matmul && p.attribute(operator_attr::matmul::kTransposeB) != 0);
  const bool transposed_a = operation == GemmOperation::Matmul && p.attribute(operator_attr::matmul::kTransposeA) != 0;
  const int columns = p.output_shape(p.output_rank() - 1);
  const int rows = linear ? p.output_count() / columns : p.output_shape(p.output_rank() - 2);
  const int shared = p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - (transposed_a ? 2 : 1));
  const int batches = linear ? 1 : p.output_count() / (rows * columns);
  float alpha = 1, beta = 0;
  if (operation == GemmOperation::Addmm) {
    alpha = attribute(p, operator_attr::addmm::kAlpha);
    beta = attribute(p, operator_attr::addmm::kBeta);
  } else if (linear && p.input_rank(KernelInput::C) >= 0) beta = 1;
  // Preserve the reference kernel's empty-inner-dimension and ignored-NaN rules.
  if (shared == 0 || alpha == 0) return false;
  const cudaDataType_t storage = gemm_storage(dtype);
  BlasContext& blas = context();
  if (beta == 1 && p.input_rank(KernelInput::C) == 1 && p.input_shape(KernelInput::C, 0) == columns &&
      p.input_dtype(KernelInput::C) == p.output_dtype() && shared > 1 && columns > 1 && batches == 1 &&
      (linear || operation == GemmOperation::Addmm)) {
    if (biased_gemm(blas, out, a, b, c, rows, columns, shared, linear, storage, alpha)) return true;
  }
  if (beta != 0) {
    if (p.input_rank(KernelInput::C) < 0 || p.input_dtype(KernelInput::C) != p.output_dtype()) return false;
    seed_addend<<<(unsigned(p.output_count()) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, c, p);
  }
  const size_t width = dtype_storage_bytes(dtype, 1);
  if (!linear && batches > 1) {
    int stride_a = batch_offset(1, KernelInput::A, p), stride_b = batch_offset(1, KernelInput::B, p);
    bool regular = true;
    for (int batch = 2; batch < batches; ++batch)
      regular = regular && batch_offset(batch, KernelInput::A, p) == batch * stride_a &&
                            batch_offset(batch, KernelInput::B, p) == batch * stride_b;
    if (regular) {
      check(cublasGemmStridedBatchedEx(blas.handle, transposed ? CUBLAS_OP_T : CUBLAS_OP_N, transposed_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                columns, rows, shared, &alpha, b, storage, transposed ? shared : columns, stride_b,
                a, storage, transposed_a ? rows : shared, stride_a, &beta, out, storage, columns,
                static_cast<long long>(rows) * columns, batches, compute32(storage),
                CUBLAS_GEMM_DEFAULT), "cublasGemmStridedBatchedEx");
      return true;
    }
  }
  for (int batch = 0; batch < batches; ++batch) {
    const uint8_t* left = reinterpret_cast<const uint8_t*>(a) + size_t(linear ? 0 : batch_offset(batch, KernelInput::A, p)) * width;
    const uint8_t* right = reinterpret_cast<const uint8_t*>(b) + size_t(linear ? 0 : batch_offset(batch, KernelInput::B, p)) * width;
    uint8_t* destination = reinterpret_cast<uint8_t*>(out) + size_t(batch) * rows * columns * width;
    // Row-major C=A*B is column-major C^T=B^T*A^T. Linear stores W as [out,in].
    check(cublasGemmEx(blas.handle, transposed ? CUBLAS_OP_T : CUBLAS_OP_N, transposed_a ? CUBLAS_OP_T : CUBLAS_OP_N,
                      columns, rows, shared, &alpha, right, storage, transposed ? shared : columns,
                      left, storage, transposed_a ? rows : shared, &beta, destination, storage, columns,
                      compute32(storage), CUBLAS_GEMM_DEFAULT), "cublasGemmEx");
  }
  return true;
}
} // namespace elman
