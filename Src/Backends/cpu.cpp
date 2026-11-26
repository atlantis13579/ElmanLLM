#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include "Core/Backend.h"
#include "Backends/HostWords.h"
#include "Kernals/integer_pointwise.h"
#include "Kernals/packed_integer_cast.h"
#include "Kernals/float64_pointwise.h"
#include "Kernals/complex_pointwise.h"
#include "Kernals/float64_reduction.h"
#include "Kernals/float64_matrix.h"
#include "Kernals/float64_softmax.h"
#include "Kernals/float64_indexing.h"
#include "Kernals/float64_spatial.h"
#include "Kernals/float64_loss.h"
#include "Kernals/float64_attention.h"
#include "Kernals/float64_random.h"
#include "Kernals/float64_categorical.h"
#include "Kernals/float64_normalization.h"
#include "Kernals/integer_matrix.h"
#include "Kernals/storage_layout.h"
#include "Kernals/low_precision_cast.h"
#include "Kernals/low_precision_pointwise.h"
#include "Kernals/low_precision_reduction.h"
#include "Kernals/low_precision_matrix.h"
#include "Kernals/low_precision_softmax.h"
#include "Kernals/low_precision_normalization.h"
#include "Kernals/low_precision_indexing.h"
#include "Kernals/low_precision_spatial.h"
#include "Kernals/low_precision_loss.h"
#include "Kernals/low_precision_fused.h"
#include "Kernals/low_precision_attention.h"
#include "Kernals/low_precision_random.h"
#include "Kernals/low_precision_categorical.h"

namespace elman {
namespace {
struct CPUBuffer final : Buffer {
  std::vector<uint64_t> storage;
  CPUBuffer(size_t count, DType dtype)
      : Buffer(Backend::CPU, dtype, count), storage(bytes / 8 + (bytes % 8 != 0)) {}
  void* raw_pointer() override { return storage.data(); }
  const void* raw_pointer() const override { return storage.data(); }
};
Buffer& cpu_buffer(const BufferPtr& value) {
  CPUBuffer* result = value ? dynamic_cast<CPUBuffer*>(&value->allocation()) : nullptr;
  if (!result) throw std::invalid_argument("Expected a CPU buffer");
  return *value;
}
} // namespace
using CPULaunch = void (*)(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
struct CPUKernel { CPULaunch launch; int inputs; };
#define ELMAN_CPU_KERNEL(name) \
  void launch_##name##_cpu(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
#define ELMAN_CPU_KERNEL4(name) \
  void launch_##name##_cpu(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
#include "Kernals/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL
#undef ELMAN_CPU_KERNEL4

BufferPtr allocate_cpu(size_t count, DType dtype) { return std::make_shared<CPUBuffer>(count, dtype); }
void clear_cpu(const BufferPtr& buffer) {
  Buffer& destination = cpu_buffer(buffer);
  if (destination.bytes) std::memset(destination.raw_pointer(), 0, destination.bytes);
}
void upload_raw_cpu(const BufferPtr& buffer, const std::vector<uint8_t>& bytes) {
  Buffer& destination = cpu_buffer(buffer);
  if (bytes.size() != destination.bytes) throw std::invalid_argument("CPU raw upload size mismatch");
  if (!bytes.empty()) std::memcpy(destination.raw_pointer(), bytes.data(), bytes.size());
}
std::vector<uint8_t> download_raw_cpu(const BufferPtr& buffer) {
  const Buffer& source = cpu_buffer(buffer);
  std::vector<uint8_t> bytes(source.bytes);
  if (!bytes.empty()) std::memcpy(bytes.data(), source.raw_pointer(), bytes.size());
  return bytes;
}
void upload_cpu(const BufferPtr& buffer, const std::vector<float>& values) {
  if (values.size() != buffer->count) throw std::invalid_argument("CPU upload size mismatch");
  upload_raw_cpu(buffer, encode_host_values(values, buffer->dtype));
}
std::vector<float> download_cpu(const BufferPtr& buffer) {
  return decode_host_values(download_raw_cpu(buffer), buffer->dtype, buffer->count);
}
void upload_int_cpu(const BufferPtr& buffer, const std::vector<int32_t>& values) {
  if (values.size() != buffer->count) throw std::invalid_argument("CPU integer upload size mismatch");
  upload_raw_cpu(buffer, encode_host_integers(values, buffer->dtype));
}
std::vector<int32_t> download_int_cpu(const BufferPtr& buffer) {
  return decode_host_integers(download_raw_cpu(buffer), buffer->dtype, buffer->count);
}
void dispatch_cpu(const std::string& operation, const BufferPtr& output,
                  const std::vector<BufferPtr>& inputs, const KernelParams& params) {
  static const std::unordered_map<std::string, CPUKernel> kernels = {
#define ELMAN_CPU_KERNEL(name) {#name, {[](uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const uint32_t*, const KernelParams& p) { launch_##name##_cpu(out, a, b, c, p); }, 3}},
#define ELMAN_CPU_KERNEL4(name) {#name, {launch_##name##_cpu, 4}},
#include "Kernals/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL
#undef ELMAN_CPU_KERNEL4
  };
  const ComplexPointwise complex_operation = complex_pointwise_operation(operation, params);
  const LowPrecisionPointwise low_operation = low_precision_pointwise_operation(operation, params);
  const Float64Pointwise double_operation = low_operation == LowPrecisionPointwise::None ? float64_pointwise_operation(operation, params) : Float64Pointwise::None;
  auto found = kernels.find(operation);
  if (found == kernels.end() && double_operation == Float64Pointwise::None && complex_operation == ComplexPointwise::None)
    throw std::invalid_argument("Unknown CPU operator: " + operation);
  const int arity = complex_operation != ComplexPointwise::None ? complex_pointwise_arity(complex_operation) :
      found == kernels.end() ? float64_pointwise_arity(double_operation) : found->second.inputs;
  Buffer& destination = cpu_buffer(output);
  if (params.v[0] < 0 || size_t(params.v[0]) != destination.count)
    throw std::invalid_argument("CPU dispatch size mismatch");
  const uint32_t* sources[4] = {nullptr, nullptr, nullptr, nullptr};
  if (inputs.size() > 4) throw std::invalid_argument("CPU kernels accept at most four inputs");
  for (size_t j = 0; j < inputs.size(); ++j) {
    if (inputs[j] && int(j) >= arity) throw std::invalid_argument("Unexpected CPU kernel input");
    if (inputs[j]) sources[j] = static_cast<const uint32_t*>(cpu_buffer(inputs[j]).raw_pointer());
  }
  const Float64Reduction double_reduction = float64_reduction_operation(operation, params);
  const Float64Matrix double_matrix = float64_matrix_operation(operation, params);
  const Float64Softmax double_softmax = float64_softmax_operation(operation, params);
  const Float64Indexing double_indexing = float64_indexing_operation(operation, params);
  const Float64Spatial double_spatial = float64_spatial_operation(operation, params);
  const Float64Loss double_loss = float64_loss_operation(operation, params);
  const Float64Attention double_attention = float64_attention_operation(operation, params);
  const Float64Random double_random = float64_random_operation(operation, params);
  const bool double_categorical = float64_categorical_operation(operation, params);
  const Float64Normalization double_normalization = float64_normalization_operation(operation, params);
  const IntegerPointwise integer_operation = integer_pointwise_operation(operation, params);
  const IntegerMatrix integer_matrix = integer_matrix_operation(operation, params);
  const StorageLayout storage_layout = storage_layout_operation(operation, output->dtype);
  if (complex_operation != ComplexPointwise::None)
      launch_complex_pointwise_cpu(complex_operation, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
    else if (packed_integer_cast_operation(operation, params))
    launch_packed_integer_cast_cpu(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], params);
  else if (low_precision_cast_operation(operation, params))
    launch_low_precision_cast_cpu(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], params);
  else if (low_precision_categorical_operation(operation, params))
    launch_low_precision_categorical_cpu(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_random_operation(operation, params) != LowPrecisionRandom::None)
    launch_low_precision_random_cpu(low_precision_random_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], params);
  else if (low_precision_attention_operation(operation, params) == LowPrecisionAttention::Matrix)
    launch_low_attention_matrix_cpu(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], params);
  else if (low_precision_attention_operation(operation, params) == LowPrecisionAttention::Sum || low_precision_attention_operation(operation, params) == LowPrecisionAttention::SumTo)
    launch_low_attention_sum_cpu(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], params);
  else if (low_precision_attention_operation(operation, params) == LowPrecisionAttention::Gradient || low_precision_attention_operation(operation, params) == LowPrecisionAttention::Probabilities)
    launch_low_precision_softmax_cpu(LowPrecisionSoftmax::Softmax, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], low_attention_softmax_parameters(params, low_precision_attention_operation(operation, params)));
  else if (low_precision_fused_operation(operation, params) != LowPrecisionFused::None)
    launch_low_precision_fused_cpu(low_precision_fused_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_loss_operation(operation, params) != LowPrecisionLoss::None)
    launch_low_precision_loss_cpu(low_precision_loss_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_spatial_operation(operation, params) != LowPrecisionSpatial::None)
    launch_low_precision_spatial_cpu(low_precision_spatial_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_indexing_operation(operation, params) != LowPrecisionIndexing::None)
    launch_low_precision_indexing_cpu(low_precision_indexing_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_normalization_operation(operation, params) != LowPrecisionNormalization::None)
    launch_low_precision_normalization_cpu(low_precision_normalization_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_softmax_operation(operation, params) != LowPrecisionSoftmax::None)
    launch_low_precision_softmax_cpu(low_precision_softmax_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], params);
  else if (low_precision_matrix_operation(operation, params) != LowPrecisionMatrix::None)
    launch_low_precision_matrix_cpu(low_precision_matrix_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (low_precision_reduction_operation(operation, params) != LowPrecisionReduction::None)
    launch_low_precision_reduction_cpu(low_precision_reduction_operation(operation, params), static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
    else if (low_operation != LowPrecisionPointwise::None)
    launch_low_precision_pointwise_cpu(low_operation, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (storage_layout != StorageLayout::None)
    launch_storage_layout_cpu(storage_layout, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_indexing != Float64Indexing::None)
    launch_float64_indexing_cpu(double_indexing, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_spatial != Float64Spatial::None)
    launch_float64_spatial_cpu(double_spatial, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_loss != Float64Loss::None)
    launch_float64_loss_cpu(double_loss, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_attention != Float64Attention::None)
    launch_float64_attention_cpu(double_attention, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], sources[3], params);
  else if (double_random != Float64Random::None)
    launch_float64_random_cpu(double_random, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], params);
  else if (double_normalization != Float64Normalization::None)
    launch_float64_normalization_cpu(double_normalization, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_categorical)
    launch_float64_categorical_cpu(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_softmax != Float64Softmax::None)
    launch_float64_softmax_cpu(double_softmax, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], params);
  else if (double_matrix != Float64Matrix::None)
    launch_float64_matrix_cpu(double_matrix, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_reduction != Float64Reduction::None)
    launch_float64_reduction_cpu(double_reduction, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (double_operation != Float64Pointwise::None)
    launch_float64_pointwise_cpu(double_operation, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (integer_operation != IntegerPointwise::None)
    launch_integer_pointwise_cpu(integer_operation, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else if (integer_matrix != IntegerMatrix::None)
    launch_integer_matrix_cpu(integer_matrix, static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], params);
  else found->second.launch(static_cast<uint32_t*>(destination.raw_pointer()), sources[0], sources[1], sources[2], sources[3], params);
}
} // namespace elman
