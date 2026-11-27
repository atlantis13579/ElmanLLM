#include "Backends/CudaConvolution.h"
#include "Backends/CudaGemm.h"

#include "Backends/CudaExecution.h"
#include "Core/AllocationPool.h"
#include "Kernals/conv2d_params.h"

#include <cuda_runtime_api.h>
#include <cudnn.h>

#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <stdexcept>
#include <utility>

namespace elman {
namespace {
using namespace kernal_attr::conv2d;
using Shape4 = std::array<int, 4>;
constexpr size_t workspace_limit = 32 * 1024 * 1024;
constexpr size_t plan_limit = 32;

void check(cudnnStatus_t status, const char* action) {
  if (status != CUDNN_STATUS_SUCCESS)
    throw std::runtime_error(std::string(action) + ": " + cudnnGetErrorString(status));
}

template <class Descriptor, cudnnStatus_t (CUDNNWINAPI *Create)(Descriptor*),
          cudnnStatus_t (CUDNNWINAPI *Destroy)(Descriptor)>
struct OwnedDescriptor {
  Descriptor value = nullptr;
  OwnedDescriptor() { check(Create(&value), "Create cuDNN descriptor"); }
  ~OwnedDescriptor() { Destroy(value); }
  OwnedDescriptor(const OwnedDescriptor&) = delete;
  OwnedDescriptor& operator=(const OwnedDescriptor&) = delete;
  operator Descriptor() const { return value; }
};

using TensorDescriptor = OwnedDescriptor<cudnnTensorDescriptor_t, cudnnCreateTensorDescriptor, cudnnDestroyTensorDescriptor>;
using FilterDescriptor = OwnedDescriptor<cudnnFilterDescriptor_t, cudnnCreateFilterDescriptor, cudnnDestroyFilterDescriptor>;
using ConvolutionDescriptor = OwnedDescriptor<cudnnConvolutionDescriptor_t, cudnnCreateConvolutionDescriptor,
                                               cudnnDestroyConvolutionDescriptor>;

struct Plan {
  TensorDescriptor input, output;
  FilterDescriptor weight;
  ConvolutionDescriptor convolution;
  int algorithm = -1;
  size_t workspace_bytes = 0;
  bool tf32 = false;

  template <class Perf, size_t N>
  void select(cudnnStatus_t status, const std::array<Perf, N>& results, int count) {
    if (status == CUDNN_STATUS_NOT_SUPPORTED) return;
    check(status, "Select cuDNN convolution algorithm");
    for (int i = 0; i < count; ++i) {
      const Perf& candidate = results[i];
      if (candidate.status != CUDNN_STATUS_SUCCESS || candidate.determinism != CUDNN_DETERMINISTIC ||
          candidate.memory > workspace_limit || candidate.mathType == CUDNN_TENSOR_OP_MATH_ALLOW_CONVERSION ||
          (!tf32 && candidate.mathType != CUDNN_FMA_MATH)) continue;
      check(cudnnSetConvolutionMathType(convolution, candidate.mathType), "Set cuDNN convolution math type");
      algorithm = static_cast<int>(candidate.algo);
      workspace_bytes = candidate.memory;
      break;
    }
  }

  Plan(cudnnHandle_t handle, CudaConvolution operation, const Shape4& x, const Shape4& w,
       const Shape4& y, const std::array<int, 7>& geometry, bool tf32_enabled) : tf32(tf32_enabled) {
    void (*const tensor)(const TensorDescriptor&, const Shape4&) = [](const TensorDescriptor& d, const Shape4& s) {
      check(cudnnSetTensor4dDescriptor(d, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, s[0], s[1], s[2], s[3]),
            "Set cuDNN tensor shape");
    };
    tensor(input, x);
    tensor(output, y);
    check(cudnnSetFilter4dDescriptor(weight, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, w[0], w[1], w[2], w[3]),
          "Set cuDNN filter shape");
    check(cudnnSetConvolution2dDescriptor(convolution, geometry[kPaddingHeight], geometry[kPaddingWidth],
          geometry[kStrideHeight], geometry[kStrideWidth], geometry[kDilationHeight], geometry[kDilationWidth],
          CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT), "Set cuDNN convolution geometry");
    check(cudnnSetConvolutionGroupCount(convolution, geometry[kGroups]), "Set cuDNN convolution groups");
    check(cudnnSetConvolutionMathType(convolution,
          tf32 ? CUDNN_DEFAULT_MATH : CUDNN_FMA_MATH),
          "Set cuDNN convolution math mode");
    int count = 0;
    switch (operation) {
      case CudaConvolution::Forward: {
        std::array<cudnnConvolutionFwdAlgoPerf_t, CUDNN_CONVOLUTION_FWD_ALGO_COUNT> results{};
        cudnnStatus_t status = cudnnGetConvolutionForwardAlgorithm_v7(handle, input, weight, convolution, output,
                                                             int(results.size()), &count, results.data());
        select(status, results, count);
        break;
      }
      case CudaConvolution::InputGradient: {
        std::array<cudnnConvolutionBwdDataAlgoPerf_t, CUDNN_CONVOLUTION_BWD_DATA_ALGO_COUNT> results{};
        cudnnStatus_t status = cudnnGetConvolutionBackwardDataAlgorithm_v7(handle, weight, output, convolution, input,
                                                                  int(results.size()), &count, results.data());
        select(status, results, count);
        break;
      }
      case CudaConvolution::WeightGradient: {
        std::array<cudnnConvolutionBwdFilterAlgoPerf_t, CUDNN_CONVOLUTION_BWD_FILTER_ALGO_COUNT> results{};
        cudnnStatus_t status = cudnnGetConvolutionBackwardFilterAlgorithm_v7(handle, input, output, convolution, weight,
                                                                    int(results.size()), &count, results.data());
        select(status, results, count);
        break;
      }
    }
  }
};

using Key = std::array<int, 20>;
struct Context {
  int device;
  cudnnHandle_t handle = nullptr;
  std::map<Key, std::unique_ptr<Plan>> plans;
  explicit Context(int ordinal) : device(ordinal) { check(cudnnCreate(&handle), "Create cuDNN handle"); }
  ~Context() {
    int previous = -1;
    cudaGetDevice(&previous);
    if (previous != device) cudaSetDevice(device);
    plans.clear();
    cudnnDestroy(handle);
    if (previous >= 0 && previous != device) cudaSetDevice(previous);
  }
};

bool aligned(const void* pointer) {
  return reinterpret_cast<uintptr_t>(pointer) % 16 == 0;
}
}  // namespace

bool try_convolution_cuda(CudaConvolution operation, uint32_t* out, const uint32_t* a,
                          const uint32_t* b, const KernelParams& p) {
  constexpr int fp32 = int(DType::Float32);
  if (p.output_dtype() != fp32 || p.input_dtype(KernelInput::A) != fp32 ||
      p.input_dtype(KernelInput::B) != fp32 ||
      !aligned(out) || !aligned(a) || !aligned(b))
    return false;
  if (p.output_rank() != 4 || p.input_rank(KernelInput::A) != 4 || p.input_rank(KernelInput::B) != 4) return false;
  Shape4 x{}, w{}, y{};
  for (int axis = 0; axis < 4; ++axis) {
    const int left = p.input_shape(KernelInput::A, axis), right = p.input_shape(KernelInput::B, axis);
    const int destination = p.output_shape(axis);
    if (left <= 0 || right <= 0 || destination <= 0) return false;
    x[axis] = operation == CudaConvolution::InputGradient ? destination : left;
    w[axis] = operation == CudaConvolution::WeightGradient ? destination : right;
    y[axis] = operation == CudaConvolution::Forward ? destination :
              operation == CudaConvolution::InputGradient ? left : right;
  }
  constexpr int64_t minimum_work = 1LL << 19;
  const int64_t work = int64_t(y[0]) * y[1] * y[2] * y[3] * w[1] * w[2] * w[3];
  if (work < minimum_work) return false;

  const cudaStream_t stream = current_cuda_stream();
  cudaStreamCaptureStatus capturing;
  check_cuda_execution(cudaStreamIsCapturing(stream, &capturing), "Query CUDA convolution stream");
  if (capturing != cudaStreamCaptureStatusNone) return false;
  int device = 0;
  check_cuda_execution(cudaGetDevice(&device), "Get CUDA convolution device");
  const bool tf32 = cuda_conv_tf32_enabled();
  static thread_local std::map<std::pair<int, bool>, std::unique_ptr<Context>> contexts;
  std::unique_ptr<Context>& owner = contexts[{device, tf32}];
  if (!owner) owner = std::make_unique<Context>(device);
  Context& context = *owner;
  check(cudnnSetStream(context.handle, stream), "Set cuDNN stream");

  std::array<int, 7> geometry{};
  Key key{};
  int position = 0;
  key[position++] = static_cast<int>(operation);
  for (const std::array<int, 4>& shape : {x, w, y}) for (int extent : shape) key[position++] = extent;
  for (size_t i = 0; i < geometry.size(); ++i) key[position++] = geometry[i] = p.attribute(int(i));
  auto entry = context.plans.find(key);
  if (entry == context.plans.end()) {
    if (context.plans.size() == plan_limit) context.plans.clear();
    entry = context.plans.emplace(key, std::make_unique<Plan>(context.handle, operation, x, w, y, geometry, tf32)).first;
  }
  const Plan& plan = *entry->second;
  if (plan.algorithm < 0) return false;
  BufferPtr workspace;
  if (plan.workspace_bytes) {
    try { workspace = backend_workspace(Backend::CUDA, plan.workspace_bytes); }
    catch (const MemoryLimitExceeded&) { return false; }
    catch (const AllocationFailure&) { return false; }
  }
  void* scratch = workspace ? workspace->raw_pointer() : nullptr;
  const float one = 1.f, zero = 0.f;
  cudnnStatus_t status;
  switch (operation) {
    case CudaConvolution::Forward:
      status = cudnnConvolutionForward(context.handle, &one, plan.input, a, plan.weight, b, plan.convolution,
          static_cast<cudnnConvolutionFwdAlgo_t>(plan.algorithm), scratch, plan.workspace_bytes, &zero, plan.output, out);
      break;
    case CudaConvolution::InputGradient:
      status = cudnnConvolutionBackwardData(context.handle, &one, plan.weight, b, plan.output, a, plan.convolution,
          static_cast<cudnnConvolutionBwdDataAlgo_t>(plan.algorithm), scratch, plan.workspace_bytes, &zero, plan.input, out);
      break;
    case CudaConvolution::WeightGradient:
      status = cudnnConvolutionBackwardFilter(context.handle, &one, plan.input, a, plan.output, b, plan.convolution,
          static_cast<cudnnConvolutionBwdFilterAlgo_t>(plan.algorithm), scratch, plan.workspace_bytes, &zero, plan.weight, out);
      break;
    default:
      throw std::invalid_argument("Unknown CUDA convolution operation");
  }
  if (status == CUDNN_STATUS_NOT_SUPPORTED) return false;
  check(status, "Execute cuDNN convolution");
  return true;
}
}  // namespace elman
