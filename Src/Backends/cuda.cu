#include <cuda_runtime.h>
#include <math_constants.h>
#include <climits>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include "Backends/HostWords.h"
#include "Core/Backend.h"
#include "Core/Memory.h"
#include "Core/Device.h"
#include "Core/CommandGraph.h"
#include "Backends/CudaExecution.h"
#include "Backends/CudaGreedy.h"
#include "Core/AllocationPool.h"
#include "Operators/integer_pointwise.h"
#include "Operators/packed_integer_cast.h"
#include "Operators/float64_pointwise.h"
#include "Operators/complex_pointwise.h"
#include "Operators/float64_reduction.h"
#include "Operators/float64_matrix.h"
#include "Operators/float64_softmax.h"
#include "Operators/float64_indexing.h"
#include "Operators/float64_spatial.h"
#include "Operators/float64_loss.h"
#include "Operators/float64_attention.h"
#include "Operators/float64_random.h"
#include "Operators/float64_categorical.h"
#include "Operators/float64_normalization.h"
#include "Operators/integer_matrix.h"
#include "Operators/storage_layout.h"
#include "Operators/low_precision_cast.h"
#include "Operators/low_precision_pointwise.h"
#include "Operators/low_precision_reduction.h"
#include "Operators/low_precision_matrix.h"
#include "Operators/low_precision_softmax.h"
#include "Operators/low_precision_normalization.h"
#include "Operators/low_precision_indexing.h"
#include "Operators/low_precision_spatial.h"
#include "Operators/low_precision_loss.h"
#include "Operators/low_precision_fused.h"
#include "Operators/low_precision_attention.h"
#include "Operators/low_precision_random.h"
#include "Operators/low_precision_categorical.h"

namespace elman {
namespace {
void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    // Consume the handled runtime error so a recoverable allocation failure
    // does not surface again at the next kernel's launch check.
    cudaGetLastError();
    if (result == cudaErrorMemoryAllocation) throw AllocationFailure(std::string(operation) + ": " + cudaGetErrorString(result));
    throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
  }
}
struct CudaBuffer final : Buffer {
  void* storage = nullptr;
  int device = 0;
  CudaAccess access;
  CudaBuffer(size_t count, DType dtype, bool initialize = true) : Buffer(Backend::CUDA, dtype, count) {
    std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
    cuda_check(cudaGetDevice(&device), "cudaGetDevice");
    cuda_check(cudaMalloc(&storage, std::max(bytes, size_t(1))), "cudaMalloc");
    try {
      if (initialize) {
        cuda_check(cudaMemsetAsync(storage, 0, std::max(bytes, size_t(1)), current_cuda_stream()), "cudaMemsetAsync");
        finish_cuda_write(access, record_cuda_completion());
      }
    } catch (...) { cudaFree(storage); throw; }
  }
  ~CudaBuffer() override {
    if (!storage) return;
    int previous = -1;
    cudaGetDevice(&previous);
    bool changed = previous != device && cudaSetDevice(device) == cudaSuccess;
    cudaFree(storage);
    if (changed && previous >= 0) cudaSetDevice(previous);
  }
  void* raw_pointer() override { return storage; }
  const void* raw_pointer() const override { return storage; }
  int device_ordinal() const override { return device; }
};
CudaBuffer& checked_cuda(const BufferPtr& buffer, int device) {
  CudaBuffer* value = buffer ? dynamic_cast<CudaBuffer*>(&buffer->allocation()) : nullptr;
  if (!value) throw std::runtime_error("Expected a CUDA buffer");
  if (device != value->device) throw std::invalid_argument("CUDA buffer belongs to another device");
  return *value;
}
CudaBuffer& checked_cuda(const BufferPtr& buffer) {
  int device = -1;
  cuda_check(cudaGetDevice(&device), "cudaGetDevice");
  return checked_cuda(buffer, device);
}
struct WorkspaceFrame;
thread_local WorkspaceFrame* workspace_frame = nullptr;
struct WorkspaceFrame {
  WorkspaceFrame* previous = workspace_frame;
  std::vector<BufferPtr> buffers;
  WorkspaceFrame() { workspace_frame = this; }
  ~WorkspaceFrame() { workspace_frame = previous; }
};
} // namespace
using CudaLaunch = void (*)(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
struct CudaKernel { CudaLaunch launch; int inputs; };
#define ELMAN_CUDA_KERNEL(name) \
  void launch_##name##_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
#define ELMAN_CUDA_KERNEL4(name) \
  void launch_##name##_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
#include "Operators/cuda_dispatch.inc"
#undef ELMAN_CUDA_KERNEL
#undef ELMAN_CUDA_KERNEL4

bool available_cuda() {
  int devices = 0;
  if (cudaGetDeviceCount(&devices) != cudaSuccess) { cudaGetLastError(); return false; }
  return devices > 0;
}
MemoryInfo memory_info_cuda() {
  MemoryInfo result;
  cuda_check(cudaMemGetInfo(&result.available_bytes, &result.total_bytes), "cudaMemGetInfo");
  return result;
}
std::vector<DeviceInfo> devices_cuda() {
  int count = 0;
  cudaError_t status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) { cudaGetLastError(); return {}; }
  cuda_check(status, "cudaGetDeviceCount");
  std::vector<DeviceInfo> result;
  for (int ordinal = 0; ordinal < count; ++ordinal) {
    cudaDeviceProp properties{};
    cuda_check(cudaGetDeviceProperties(&properties, ordinal), "cudaGetDeviceProperties");
    result.push_back({Backend::CUDA, ordinal, properties.name, properties.totalGlobalMem});
  }
  return result;
}
void select_cuda_device(int ordinal) {
  if (ordinal < 0) throw std::invalid_argument("CUDA device ordinal must be nonnegative");
  cuda_check(cudaSetDevice(ordinal), "cudaSetDevice");
}
void copy_cuda_storage(const BufferPtr& destination, const BufferPtr& source) {
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  CudaBuffer& target = checked_cuda(destination);
  CudaBuffer* origin = dynamic_cast<CudaBuffer*>(&source->allocation());
  if (!origin) throw std::invalid_argument("CUDA transfer source is not device storage");
  if (!destination->bytes) return;
  if (destination->raw_pointer() == source->raw_pointer() && target.device == origin->device) return;
  if (&destination->allocation() == &source->allocation()) {
    const uintptr_t left = reinterpret_cast<uintptr_t>(destination->raw_pointer());
    const uintptr_t right = reinterpret_cast<uintptr_t>(source->raw_pointer());
    if (left <= right ? right - left < destination->bytes : left - right < source->bytes)
      throw std::invalid_argument("CUDA transfer storage overlaps");
  }
  prepare_cuda_write(target.access);
  try {
    int peer = target.device == origin->device;
    if (!peer) cuda_check(cudaDeviceCanAccessPeer(&peer, target.device, origin->device), "cudaDeviceCanAccessPeer");
    if (peer) {
      prepare_cuda_read(origin->access);
      if (target.device == origin->device)
        cuda_check(cudaMemcpyAsync(destination->raw_pointer(), source->raw_pointer(), destination->bytes,
                                   cudaMemcpyDeviceToDevice, current_cuda_stream()), "CUDA device copy");
      else {
        cudaError_t status = cudaDeviceEnablePeerAccess(origin->device, 0);
        if (status == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        else cuda_check(status, "cudaDeviceEnablePeerAccess");
        cuda_check(cudaMemcpyPeerAsync(destination->raw_pointer(), target.device, source->raw_pointer(),
                                       origin->device, destination->bytes, current_cuda_stream()), "CUDA peer copy");
      }
      if (!asynchronous_execution(Backend::CUDA)) synchronize_cuda_stream();
      std::shared_ptr<CudaEventState> event = record_cuda_completion();
      finish_cuda_read(origin->access, event);
      finish_cuda_write(target.access, event);
    } else {
      const size_t capacity = std::min(destination->bytes, size_t(4 * 1024 * 1024));
      MemoryReservation host(Backend::CPU, capacity);
      std::vector<uint8_t> staging(capacity);
      for (size_t offset = 0; offset < destination->bytes; offset += capacity) {
        const size_t count = std::min(capacity, destination->bytes - offset);
        {
          DeviceScope owner(Backend::CUDA, origin->device);
          prepare_cuda_read(origin->access);
          cuda_check(cudaMemcpyAsync(staging.data(), static_cast<const uint8_t*>(source->raw_pointer()) + offset,
              count, cudaMemcpyDeviceToHost, current_cuda_stream()), "CUDA staged source copy");
          synchronize_cuda_stream();
        }
        cuda_check(cudaMemcpyAsync(static_cast<uint8_t*>(destination->raw_pointer()) + offset, staging.data(),
            count, cudaMemcpyHostToDevice, current_cuda_stream()), "CUDA staged destination copy");
        synchronize_cuda_stream();
      }
      finish_cuda_write(target.access, {});
    }
  } catch (...) { cudaStreamSynchronize(current_cuda_stream()); throw; }
}
BufferPtr allocate_cuda(size_t count, DType dtype) { return std::make_shared<CudaBuffer>(count, dtype); }
BufferPtr allocate_workspace_cuda(size_t count, DType dtype) { return std::make_shared<CudaBuffer>(count, dtype, false); }
int allocation_device_cuda() {
  int device = 0;
  cuda_check(cudaGetDevice(&device), "cudaGetDevice");
  return device;
}
void clear_cuda(const BufferPtr& buffer) {
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  CudaBuffer& destination = checked_cuda(buffer);
  prepare_cuda_write(destination.access);
  try {
    if (buffer->bytes) cuda_check(cudaMemsetAsync(buffer->raw_pointer(), 0, buffer->bytes, current_cuda_stream()), "CUDA clear reused buffer");
    finish_cuda_write(destination.access, record_cuda_completion());
  } catch (...) { cudaStreamSynchronize(current_cuda_stream()); throw; }
}
void retain_cuda_workspace(const BufferPtr& buffer) {
  if (detail::lowering_cuda_graph()) return;
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  prepare_cuda_write(checked_cuda(buffer).access);
  if (workspace_frame) workspace_frame->buffers.push_back(buffer);
}
void upload_raw_cuda(const BufferPtr& buffer, const std::vector<uint8_t>& bytes) {
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  CudaBuffer& destination = checked_cuda(buffer);
  if (bytes.size() != buffer->bytes) throw std::invalid_argument("CUDA raw upload size mismatch");
  prepare_cuda_write(destination.access);
  try {
    if (!bytes.empty()) if (std::shared_ptr<CudaEventState> completion = stage_cuda_upload(buffer->raw_pointer(), bytes.data(), bytes.size())) {
      finish_cuda_write(destination.access, completion);
      return;
    }
    if (!bytes.empty()) cuda_check(cudaMemcpyAsync(buffer->raw_pointer(), bytes.data(), bytes.size(), cudaMemcpyHostToDevice, current_cuda_stream()), "CUDA raw upload");
    // Pageable caller storage can be destroyed immediately after this call.
    synchronize_cuda_stream();
    finish_cuda_write(destination.access, {});
  } catch (...) { cudaStreamSynchronize(current_cuda_stream()); throw; }
}
std::vector<uint8_t> download_raw_cuda(const BufferPtr& buffer) {
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  const CudaBuffer& source = checked_cuda(buffer);
  std::vector<uint8_t> bytes(buffer->bytes);
  prepare_cuda_read(source.access);
  try {
    if (!bytes.empty() && stage_cuda_download(bytes.data(), buffer->raw_pointer(), bytes.size())) return bytes;
    if (!bytes.empty()) cuda_check(cudaMemcpyAsync(bytes.data(), buffer->raw_pointer(), bytes.size(), cudaMemcpyDeviceToHost, current_cuda_stream()), "CUDA raw download");
    synchronize_cuda_stream();
  } catch (...) { cudaStreamSynchronize(current_cuda_stream()); throw; }
  return bytes;
}
void upload_cuda(const BufferPtr& buffer, const std::vector<float>& values) {
  if (values.size() != buffer->count) throw std::invalid_argument("CUDA upload size mismatch");
  upload_raw_cuda(buffer, encode_host_values(values, buffer->dtype));
}
std::vector<float> download_cuda(const BufferPtr& buffer) {
  return decode_host_values(download_raw_cuda(buffer), buffer->dtype, buffer->count);
}
namespace {
struct Candidate {
  float value;
  int id;
};
struct FirstMaximum {
  __device__ Candidate operator()(Candidate a, Candidate b) const {
    if (a.id < 0) return a;
    if (b.id < 0) return b;
    return b.value > a.value || (b.value == a.value && b.id < a.id) ? b : a;
  }
};
constexpr int greedy_threads = 256;
__device__ Candidate block_maximum(Candidate value) {
  __shared__ Candidate lanes[greedy_threads];
  lanes[threadIdx.x] = value;
  __syncthreads();
  for (int stride = greedy_threads / 2; stride; stride /= 2) {
    if (threadIdx.x < stride)
      lanes[threadIdx.x] = FirstMaximum{}(lanes[threadIdx.x], lanes[threadIdx.x + stride]);
    __syncthreads();
  }
  return lanes[0];
}
__global__ void greedy_blocks(const float* logits, size_t count, Candidate* partials) {
  Candidate best{-CUDART_INF_F, INT_MAX};
  for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x; i < count; i += gridDim.x * blockDim.x) {
    float value = logits[i];
    best = FirstMaximum{}(best, {value, isfinite(value) ? int(i) : -1});
  }
  best = block_maximum(best);
  if (threadIdx.x == 0) partials[blockIdx.x] = best;
}
__global__ void greedy_final(const Candidate* partials, int count, int* result) {
  Candidate best = threadIdx.x < count ? partials[threadIdx.x] : Candidate{-CUDART_INF_F, INT_MAX};
  best = block_maximum(best);
  if (threadIdx.x == 0) *result = best.id;
}
}
int32_t greedy_id_cuda(const BufferPtr& logits) {
  if (logits->dtype != DType::Float32 || !logits->count || logits->count > INT_MAX)
    throw std::invalid_argument("CUDA greedy selection requires a nonempty Float32 vocabulary");
  const int blocks = int(std::min<size_t>(greedy_threads, (logits->count + greedy_threads - 1) / greedy_threads));
  DeviceScope owner(Backend::CUDA, logits->device_ordinal());
  BufferPtr workspace = backend_workspace(Backend::CUDA, blocks * sizeof(Candidate) + sizeof(int));
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  prepare_cuda_read(checked_cuda(logits).access);
  Candidate* partials = static_cast<Candidate*>(workspace->raw_pointer());
  int* result = reinterpret_cast<int*>(partials + blocks);
  int id;
  try {
    greedy_blocks<<<blocks, greedy_threads, 0, current_cuda_stream()>>>(
        static_cast<const float*>(logits->raw_pointer()), logits->count, partials);
    greedy_final<<<1, greedy_threads, 0, current_cuda_stream()>>>(partials, blocks, result);
    cuda_check(cudaGetLastError(), "CUDA greedy selection");
    cuda_check(cudaMemcpyAsync(&id, result, sizeof(id), cudaMemcpyDeviceToHost, current_cuda_stream()), "CUDA greedy ID");
    synchronize_cuda_stream();
  } catch (...) { cudaStreamSynchronize(current_cuda_stream()); throw; }
  return id;
}
void upload_int_cuda(const BufferPtr& buffer, const std::vector<int32_t>& values) {
  if (values.size() != buffer->count) throw std::invalid_argument("CUDA integer upload size mismatch");
  upload_raw_cuda(buffer, encode_host_integers(values, buffer->dtype));
}
std::vector<int32_t> download_int_cuda(const BufferPtr& buffer) {
  return decode_host_integers(download_raw_cuda(buffer), buffer->dtype, buffer->count);
}
void dispatch_cuda(const std::string& op, const BufferPtr& output,
                   const std::vector<BufferPtr>& inputs, const KernelParams& params) {
  std::lock_guard<std::recursive_mutex> lock(cuda_submission_mutex());
  static const std::unordered_map<std::string, CudaKernel> kernels = {
#define ELMAN_CUDA_KERNEL(name) {#name, {[](uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t*, const KernelParams& p) { launch_##name##_cuda(out, a, b, c, p); }, 3}},
#define ELMAN_CUDA_KERNEL4(name) {#name, {launch_##name##_cuda, 4}},
#include "Operators/cuda_dispatch.inc"
#undef ELMAN_CUDA_KERNEL
#undef ELMAN_CUDA_KERNEL4
  };
  const ComplexPointwise complex_operation = complex_pointwise_operation(op, params);
  const LowPrecisionPointwise low_operation = low_precision_pointwise_operation(op, params);
  const Float64Pointwise double_operation = low_operation == LowPrecisionPointwise::None ? float64_pointwise_operation(op, params) : Float64Pointwise::None;
  auto found = kernels.find(op);
  if (found == kernels.end() && double_operation == Float64Pointwise::None && complex_operation == ComplexPointwise::None)
    throw std::runtime_error("Unknown CUDA operator: " + op);
  const int arity = complex_operation != ComplexPointwise::None ? complex_pointwise_arity(complex_operation) :
      found == kernels.end() ? float64_pointwise_arity(double_operation) : found->second.inputs;
  CudaBuffer& target = checked_cuda(output);
  if (inputs.size() > 4) throw std::invalid_argument("CUDA kernels accept at most four inputs");
  if (params.v[0] < 0 || size_t(params.v[0]) != output->count) throw std::invalid_argument("CUDA dispatch output count mismatch");
  uint32_t* out = static_cast<uint32_t*>(output->raw_pointer());
  const uint32_t* sources[4] = {out, out, out, out};
  CudaBuffer* operands[4]{};
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i] && int(i) >= arity) throw std::invalid_argument("Unexpected CUDA kernel input");
    if (inputs[i]) {
      operands[i] = &checked_cuda(inputs[i], target.device);
      sources[i] = static_cast<const uint32_t*>(inputs[i]->raw_pointer());
    }
  }
  if (!output->count) return;
  detail::CudaCommandRecording command(op, output, inputs, params);
  if (!detail::lowering_cuda_graph()) {
    prepare_cuda_write(target.access);
    for (CudaBuffer* operand : operands) if (operand && operand != &target) prepare_cuda_read(operand->access);
  }
  WorkspaceFrame workspace;
  try {
    const Float64Reduction double_reduction = float64_reduction_operation(op, params);
    const Float64Matrix double_matrix = float64_matrix_operation(op, params);
    const Float64Softmax double_softmax = float64_softmax_operation(op, params);
    const Float64Indexing double_indexing = float64_indexing_operation(op, params);
    const Float64Spatial double_spatial = float64_spatial_operation(op, params);
    const Float64Loss double_loss = float64_loss_operation(op, params);
    const Float64Attention double_attention = float64_attention_operation(op, params);
    const Float64Random double_random = float64_random_operation(op, params);
  const bool double_categorical = float64_categorical_operation(op, params);
  const Float64Normalization double_normalization = float64_normalization_operation(op, params);
    const IntegerPointwise integer_operation = integer_pointwise_operation(op, params);
    const IntegerMatrix integer_matrix = integer_matrix_operation(op, params);
    const StorageLayout storage_layout = storage_layout_operation(op, output->dtype);
    if (complex_operation != ComplexPointwise::None)
      launch_complex_pointwise_cuda(complex_operation, out, sources[0], sources[1], sources[2], params);
    else if (packed_integer_cast_operation(op, params))
      launch_packed_integer_cast_cuda(out, sources[0], params);
    else if (low_precision_cast_operation(op, params))
      launch_low_precision_cast_cuda(out, sources[0], params);
  else if (low_precision_categorical_operation(op, params))
    launch_low_precision_categorical_cuda(out, sources[0], sources[1], sources[2], params);
  else if (low_precision_random_operation(op, params) != LowPrecisionRandom::None)
    launch_low_precision_random_cuda(low_precision_random_operation(op, params), out, sources[0], params);
  else if (low_precision_attention_operation(op, params) == LowPrecisionAttention::Matrix)
    launch_low_attention_matrix_cuda(out, sources[0], sources[1], params);
  else if (low_precision_attention_operation(op, params) == LowPrecisionAttention::Sum || low_precision_attention_operation(op, params) == LowPrecisionAttention::SumTo)
    launch_low_attention_sum_cuda(out, sources[0], params);
  else if (low_precision_attention_operation(op, params) == LowPrecisionAttention::Gradient || low_precision_attention_operation(op, params) == LowPrecisionAttention::Probabilities)
    launch_low_precision_softmax_cuda(LowPrecisionSoftmax::Softmax, out, sources[0], sources[1], low_attention_softmax_parameters(params, low_precision_attention_operation(op, params)));
  else if (low_precision_fused_operation(op, params) != LowPrecisionFused::None)
    launch_low_precision_fused_cuda(low_precision_fused_operation(op, params), out, sources[0], sources[1], sources[2], params);
  else if (low_precision_loss_operation(op, params) != LowPrecisionLoss::None)
    launch_low_precision_loss_cuda(low_precision_loss_operation(op, params), out, sources[0], sources[1], sources[2], params);
  else if (low_precision_spatial_operation(op, params) != LowPrecisionSpatial::None)
    launch_low_precision_spatial_cuda(low_precision_spatial_operation(op, params), out, sources[0], sources[1], sources[2], params);
  else if (low_precision_indexing_operation(op, params) != LowPrecisionIndexing::None)
    launch_low_precision_indexing_cuda(low_precision_indexing_operation(op, params), out, sources[0], sources[1], sources[2], params);
  else if (low_precision_normalization_operation(op, params) != LowPrecisionNormalization::None)
    launch_low_precision_normalization_cuda(low_precision_normalization_operation(op, params), out, sources[0], sources[1], sources[2], params);
  else if (low_precision_softmax_operation(op, params) != LowPrecisionSoftmax::None)
    launch_low_precision_softmax_cuda(low_precision_softmax_operation(op, params), out, sources[0], sources[1], params);
  else if (low_precision_matrix_operation(op, params) != LowPrecisionMatrix::None)
    launch_low_precision_matrix_cuda(low_precision_matrix_operation(op, params), out, sources[0], sources[1], sources[2], params);
  else if (low_precision_reduction_operation(op, params) != LowPrecisionReduction::None)
    launch_low_precision_reduction_cuda(low_precision_reduction_operation(op, params), out, sources[0], sources[1], sources[2], params);
    else if (low_operation != LowPrecisionPointwise::None)
      launch_low_precision_pointwise_cuda(low_operation, out, sources[0], sources[1], sources[2], params);
    else if (storage_layout != StorageLayout::None)
      launch_storage_layout_cuda(storage_layout, out, sources[0], sources[1], sources[2], params);
    else if (double_indexing != Float64Indexing::None)
      launch_float64_indexing_cuda(double_indexing, out, sources[0], sources[1], sources[2], params);
    else if (double_spatial != Float64Spatial::None)
      launch_float64_spatial_cuda(double_spatial, out, sources[0], sources[1], sources[2], params);
    else if (double_loss != Float64Loss::None)
      launch_float64_loss_cuda(double_loss, out, sources[0], sources[1], sources[2], params);
    else if (double_attention != Float64Attention::None)
      launch_float64_attention_cuda(double_attention, out, sources[0], sources[1], sources[2], sources[3], params);
    else if (double_random != Float64Random::None)
      launch_float64_random_cuda(double_random, out, sources[0], params);
    else if (double_normalization != Float64Normalization::None)
    launch_float64_normalization_cuda(double_normalization, out, sources[0], sources[1], sources[2], params);
  else if (double_categorical)
    launch_float64_categorical_cuda(out, sources[0], sources[1], sources[2], params);
  else if (double_softmax != Float64Softmax::None)
      launch_float64_softmax_cuda(double_softmax, out, sources[0], params);
    else if (double_matrix != Float64Matrix::None)
      launch_float64_matrix_cuda(double_matrix, out, sources[0], sources[1], sources[2], params);
    else if (double_reduction != Float64Reduction::None)
      launch_float64_reduction_cuda(double_reduction, out, sources[0], sources[1], sources[2], params);
    else if (double_operation != Float64Pointwise::None)
      launch_float64_pointwise_cuda(double_operation, out, sources[0], sources[1], sources[2], params);
    else if (integer_operation != IntegerPointwise::None)
      launch_integer_pointwise_cuda(integer_operation, out, sources[0], sources[1], sources[2], params);
    else if (integer_matrix != IntegerMatrix::None)
      launch_integer_matrix_cuda(integer_matrix, out, sources[0], sources[1], sources[2], params);
    else found->second.launch(out, sources[0], sources[1], sources[2], sources[3], params);
    cuda_check(cudaGetLastError(), ("CUDA launch " + op).c_str());
    if (detail::lowering_cuda_graph()) return;
    std::shared_ptr<CudaEventState> event = asynchronous_execution(Backend::CUDA) ? record_cuda_completion() : std::shared_ptr<CudaEventState>{};
    if (!event) synchronize_cuda_stream();
    for (CudaBuffer* operand : operands) if (operand && operand != &target) finish_cuda_read(operand->access, event);
    finish_cuda_write(target.access, event);
    for (const BufferPtr& buffer : workspace.buffers) finish_cuda_write(checked_cuda(buffer).access, event);
  } catch (...) {
    // Capture errors are unwound by the graph owner; synchronizing would itself
    // invalidate the capture before it can be ended.
    if (!detail::lowering_cuda_graph()) cudaStreamSynchronize(current_cuda_stream());
    throw;
  }
}
void prepare_cuda_graph_buffers(const std::vector<BufferPtr>& buffers) {
  for (const BufferPtr& buffer : buffers) prepare_cuda_write(checked_cuda(buffer).access);
}
void finish_cuda_graph_buffers(const std::vector<BufferPtr>& buffers) {
  const std::shared_ptr<CudaEventState> event = record_cuda_completion();
  for (const BufferPtr& buffer : buffers) finish_cuda_write(checked_cuda(buffer).access, event);
}
} // namespace elman
