#include "Backend.h"
#include "Core/LocalFunction.h"
#include "Core/TensorDispatch.h"
#include "Backends/MetalDTypeSupport.h"
#include "Core/CommandGraph.h"
#include "Operators/complex_pointwise.h"
#include "Operators/paged_attention.h"
#include "Operators/flux2_ops.h"
#include "Operators/affine_quantization.h"
#include "Core/Memory.h"
#ifdef ELMAN_HAS_METAL
#include "Backends/Metal.h"
#endif
#include "Core/AllocationPool.h"
#include "Operators/attributes.h"
#include "Operators/group_norm_params.h"
#include "Operators/integer_pointwise.h"
#include "Operators/packed_integer_cast.h"
#include "Operators/float64_pointwise.h"
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
#include "Operators/low_precision_optimizer.h"
#include "Operators/floating_dropout.h"
#include "Operators/floating_random.h"

#include <array>
#include <algorithm>
#include <atomic>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <unordered_map>

namespace elman {
BufferPtr allocate_cpu(size_t, DType);
void clear_cpu(const BufferPtr&);
void upload_cpu(const BufferPtr&, const std::vector<float>&);
std::vector<float> download_cpu(const BufferPtr&);
void upload_int_cpu(const BufferPtr&, const std::vector<int32_t>&);
std::vector<int32_t> download_int_cpu(const BufferPtr&);
void upload_raw_cpu(const BufferPtr&, const std::vector<uint8_t>&);
std::vector<uint8_t> download_raw_cpu(const BufferPtr&);
void dispatch_cpu(const std::string&, const BufferPtr&,
                  const std::vector<BufferPtr>&, const KernelParams&);

#ifdef ELMAN_HAS_CUDA
BufferPtr allocate_cuda(size_t, DType);
BufferPtr allocate_workspace_cuda(size_t, DType);
void retain_cuda_workspace(const BufferPtr&);
void clear_cuda(const BufferPtr&);
int allocation_device_cuda();
void upload_cuda(const BufferPtr&, const std::vector<float>&);
std::vector<float> download_cuda(const BufferPtr&);
void upload_int_cuda(const BufferPtr&, const std::vector<int32_t>&);
std::vector<int32_t> download_int_cuda(const BufferPtr&);
void upload_raw_cuda(const BufferPtr&, const std::vector<uint8_t>&);
std::vector<uint8_t> download_raw_cuda(const BufferPtr&);
void dispatch_cuda(const std::string&, const BufferPtr&,
                   const std::vector<BufferPtr>&, const KernelParams&);
bool available_cuda();
#endif

namespace {
struct BackendOps {
  Backend id;
  const char* name;
  bool (*available)();
  BufferPtr (*allocate)(size_t, DType);
  void (*upload)(const BufferPtr&, const std::vector<float>&);
  std::vector<float> (*download)(const BufferPtr&);
  void (*dispatch)(const std::string&, const BufferPtr&,
                   const std::vector<BufferPtr>&, const KernelParams&);
  void (*upload_int)(const BufferPtr&, const std::vector<int32_t>&);
  std::vector<int32_t> (*download_int)(const BufferPtr&);
  void (*upload_raw)(const BufferPtr&, const std::vector<uint8_t>&);
  std::vector<uint8_t> (*download_raw)(const BufferPtr&);
  void (*clear)(const BufferPtr&);
  int (*device)();
  BufferPtr (*workspace)(size_t, DType);
};

bool available_cpu() {
  return true;
}

// Add a backend by providing its native functions and one entry here.
const std::array<BackendOps, 3> implementations{{
    {Backend::CPU, "cpu", available_cpu, allocate_cpu, upload_cpu, download_cpu,
     dispatch_cpu, upload_int_cpu, download_int_cpu, upload_raw_cpu, download_raw_cpu, clear_cpu, nullptr, allocate_cpu},
#ifdef ELMAN_HAS_CUDA
    {Backend::CUDA, "cuda", available_cuda, allocate_cuda, upload_cuda,
     download_cuda, dispatch_cuda, upload_int_cuda, download_int_cuda, upload_raw_cuda, download_raw_cuda, clear_cuda, allocation_device_cuda, allocate_workspace_cuda},
#else
    {Backend::CUDA, "cuda", nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
#ifdef ELMAN_HAS_METAL
    {Backend::Metal, "metal", available_metal, allocate_metal, upload_metal, download_metal,
     dispatch_metal, upload_int_metal, download_int_metal, upload_raw_metal, download_raw_metal, clear_metal, allocation_device_metal, allocate_metal},
#else
    {Backend::Metal, "metal"},
#endif
}};

size_t backend_index(Backend backend) {
  for (size_t index = 0; index < implementations.size(); ++index) {
    if (implementations[index].id == backend) {
      return index;
    }
  }
  throw std::invalid_argument("Invalid backend");
}

const BackendOps& find_backend(Backend backend) {
  return implementations[backend_index(backend)];
}

std::array<std::atomic<uint64_t>, implementations.size()> counts{};
}  // namespace

std::string backend_name(Backend backend) {
  return find_backend(backend).name;
}

Backend parse_backend(const std::string& name) {
  for (const BackendOps& implementation : implementations) {
    if (name == implementation.name) {
      return implementation.id;
    }
  }
  throw std::invalid_argument("Unknown backend: " + name);
}

bool backend_available(Backend backend) {
  const BackendOps& implementation = find_backend(backend);
  return implementation.available && implementation.available();
}

BufferPtr backend_allocate(Backend backend, size_t count, DType dtype) {
  dtype_info(dtype);
  const BackendOps& implementation = find_backend(backend);
  if (!implementation.allocate) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  return detail::pooled_allocate(backend, count, dtype, MemoryKind::Tensor, implementation.allocate, implementation.clear);
}
BufferPtr detail::allocate_dispatch_output(const std::string& op, Backend backend, size_t count, DType dtype,
                                           const std::vector<Tensor>& inputs) {
#ifdef ELMAN_HAS_CUDA
  if (backend == Backend::CUDA && (is_arithmetic_float(dtype) || is_complex(dtype) ||
                                   dtype == DType::Bool || dtype == DType::Int32)) {
    // These kernels write every output element. Keep zero-initialized storage for
    // accumulation kernels, packed formats, and raw calls with missing operands.
    static const std::unordered_map<std::string_view, size_t> full_writes = {
      {"abs",1},{"add",2},{"cast",1},{"copy",1},{"cos",1},{"div",2},{"dropout",1},{"eq",2},
      {"exp",1},{"ge",2},{"gelu",1},{"gt",2},{"le",2},{"leaky_relu",1},{"log",1},{"lt",2},
      {"logical_and",2},{"logical_not",1},{"logical_or",2},{"maximum",2},{"minimum",2},{"mul",2},
      {"ne",2},{"normal",0},{"pow",2},{"relu",1},{"rsqrt",1},{"sign",1},{"silu",1},{"silu_mul",2},
      {"sin",1},{"sqrt",1},{"sub",2},{"swish",1},{"uniform",0},{"where",3},
      {"amax",1},{"amin",1},{"attention_softmax",2},{"log_softmax",1},{"max",1},{"mean",1},
      {"min",1},{"prod",1},{"softmax",1},{"sum",1},{"expand",1},{"gather",2},{"permute",1},
      {"repeat",1},{"reshape",1},{"slice",1},{"split",1},{"squeeze",1},{"transpose",1},
      {"tril",1},{"triu",1},{"unsqueeze",1},{"argsort",1},{"avg_pool2d",1},{"max_pool2d",1},
      {"concat",2},{"index_select",2},{"index_copy",3},{"scatter_source_grad",2},{"cumsum",1},
      {"cumsum_backward",1},{"conv2d",3},{"conv2d_input_grad",2},{"conv2d_weight_grad",2},
      {"addmm",3},{"linear",2},{"packed_linear",2},{"matmul",2},{"bmm",2},
      {"layer_norm",3},{"batch_norm",3},{"rms_norm",2},{"rotary_embedding",3},{"tiled_attention",4},
      {"complex",2},{"conj",1},{"real",1},{"imag",1},{"angle",1},{"sgn",1},{"tanh",1},
      {"float64_fma",3}
    };
    const auto it = full_writes.find(op);
    if (it != full_writes.end()) {
      const bool optional_bias = op == "linear" || op == "packed_linear";
      const uint64_t required = it->second;
      if ((inputs.size() == required || (optional_bias && inputs.size() == required + 1)) &&
          std::all_of(inputs.begin(), inputs.begin() + required, [](const Tensor& t) {
            return t.defined() && t.numel() != 0;
          }) &&
          (inputs.size() == required || !inputs.back().defined() || inputs.back().numel() != 0)) {
        return pooled_allocate(backend, count, dtype, MemoryKind::Tensor, allocate_workspace_cuda, nullptr);
      }
    }
  }
#endif
  return backend_allocate(backend, count, dtype);
}
BufferPtr backend_view(const BufferPtr& source, size_t offset, size_t count) {
  if (!source || offset > source->count || count > source->count - offset)
    throw std::invalid_argument("Contiguous view exceeds source storage");
  unsigned bits = dtype_info(source->dtype).bits;
  if ((offset % 8) * bits % 8 || ((count % 8) * bits % 8 && count != source->count - offset))
    throw std::invalid_argument("Packed view boundaries must be byte aligned or end at the source boundary");
  struct View final : Buffer {
    View(const BufferPtr& source, size_t count, size_t offset) : Buffer(source, count, offset) {}
    void* raw_pointer() override {
      void* data = allocation().raw_pointer();
      return storage_offset_bytes() ? static_cast<uint8_t*>(data) + storage_offset_bytes() : data;
    }
    const void* raw_pointer() const override {
      const void* data = allocation().raw_pointer();
      return storage_offset_bytes() ? static_cast<const uint8_t*>(data) + storage_offset_bytes() : data;
    }
  };
  return std::make_shared<View>(source, count, dtype_storage_bytes(source->dtype, offset));
}
BufferPtr backend_workspace(Backend backend, size_t bytes) {
#ifdef ELMAN_HAS_CUDA
  if (backend == Backend::CUDA) if (BufferPtr recorded = detail::replay_cuda_graph_workspace(bytes)) return recorded;
#endif
  const BackendOps& implementation = find_backend(backend);
  if (!implementation.allocate) throw std::runtime_error("Workspace backend not compiled");
  BufferPtr buffer = detail::pooled_allocate(backend, bytes, DType::UInt8, MemoryKind::Workspace, implementation.workspace, implementation.clear);
#ifdef ELMAN_HAS_CUDA
  if (backend == Backend::CUDA) {
    detail::record_cuda_graph_workspace(buffer);
    retain_cuda_workspace(buffer);
  }
#endif
  return buffer;
}
int detail::allocation_device(Backend backend) {
  const BackendOps& implementation = find_backend(backend);
  if (!implementation.allocate) throw std::runtime_error("Allocation backend not compiled");
  return implementation.device ? implementation.device() : -1;
}

void backend_upload(const BufferPtr& buffer, const std::vector<float>& values) {
  if (!buffer || values.size() != buffer->count) {
    throw std::invalid_argument("Upload size mismatch");
  }
  const BackendOps& implementation = find_backend(buffer->backend);
  if (!implementation.upload) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  implementation.upload(buffer, values);
}

std::vector<float> backend_download(const BufferPtr& buffer) {
  if (!buffer) {
    throw std::invalid_argument("Undefined buffer");
  }
  const BackendOps& implementation = find_backend(buffer->backend);
  if (!implementation.download) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  return implementation.download(buffer);
}

void backend_upload_int(const BufferPtr& buffer, const std::vector<int32_t>& values) {
  if (!buffer || (buffer->dtype != DType::Int32 && buffer->dtype != DType::Bool && !is_packed_integer(buffer->dtype)) || values.size() != buffer->count) {
    throw std::invalid_argument("Integer upload requires a matching integer or bool buffer");
  }
  const BackendOps& implementation = find_backend(buffer->backend);
  if (!implementation.upload_int) throw std::runtime_error("Integer upload backend unavailable");
  implementation.upload_int(buffer, values);
}

std::vector<int32_t> backend_download_int(const BufferPtr& buffer) {
  if (!buffer || (buffer->dtype != DType::Int32 && buffer->dtype != DType::Bool && !is_packed_integer(buffer->dtype))) {
    throw std::invalid_argument("Integer download requires an integer or bool buffer");
  }
  const BackendOps& implementation = find_backend(buffer->backend);
  if (!implementation.download_int) throw std::runtime_error("Integer download backend unavailable");
  return implementation.download_int(buffer);
}

void backend_upload_raw(const BufferPtr& buffer, const std::vector<uint8_t>& bytes) {
  if (!buffer || bytes.size() != buffer->bytes) throw std::invalid_argument("Raw upload size mismatch");
  void (*function)(const BufferPtr&, const std::vector<uint8_t>&) = find_backend(buffer->backend).upload_raw;
  if (!function) throw std::runtime_error("Raw upload backend unavailable");
  function(buffer, bytes);
}
void backend_upload_int64(const BufferPtr& buffer, const std::vector<int64_t>& values) {
  if (!buffer || buffer->dtype != DType::Int64 || values.size() != buffer->count)
    throw std::invalid_argument("Int64 upload requires matching Int64 storage");
  std::vector<uint8_t> bytes(dtype_storage_bytes(DType::Int64, values.size()));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  backend_upload_raw(buffer, bytes);
}
std::vector<int64_t> backend_download_int64(const BufferPtr& buffer) {
  if (!buffer || buffer->dtype != DType::Int64)
    throw std::invalid_argument("Int64 download requires Int64 storage");
  std::vector<uint8_t> bytes = backend_download_raw(buffer);
  if (bytes.size() != dtype_storage_bytes(DType::Int64, buffer->count))
    throw std::invalid_argument("Int64 download size mismatch");
  std::vector<int64_t> values(buffer->count);
  if (!bytes.empty()) std::memcpy(values.data(), bytes.data(), bytes.size());
  return values;
}
std::vector<uint8_t> backend_download_raw(const BufferPtr& buffer) {
  if (!buffer) throw std::invalid_argument("Undefined buffer");
  std::vector<uint8_t> (*function)(const BufferPtr&) = find_backend(buffer->backend).download_raw;
  if (!function) throw std::runtime_error("Raw download backend unavailable");
  return function(buffer);
}
void backend_dispatch(const std::string& op, const BufferPtr& output,
                      const std::vector<BufferPtr>& inputs,
                      const KernelParams& params) {
  if (!output) {
    throw std::invalid_argument("Undefined output");
  }
  if (output->count > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument(op + ": output element count exceeds Int32 indexing");
  if (output->backend == Backend::Metal) metal::validate_operator_dtypes(op, params);
  if (op == "paged_attention") {
    validate_paged_attention(output, inputs, params);
    const BackendOps& implementation = find_backend(output->backend);
    if (!implementation.dispatch) throw std::runtime_error("Paged attention backend is unavailable");
    implementation.dispatch(op, output, inputs, params);
    ++counts[backend_index(output->backend)];
    return;
  }
  const ComplexPointwise complex_operation = complex_pointwise_operation(op, params);
  if (complex_operation != ComplexPointwise::None) {
    validate_complex_pointwise(complex_operation, output, inputs, params);
    const BackendOps& implementation = find_backend(output->backend);
    if (!implementation.dispatch) throw std::runtime_error("Complex backend is unavailable");
    implementation.dispatch(op, output, inputs, params);
    ++counts[backend_index(output->backend)];
    return;
  }
  const AffineQuantization quantization = affine_quantization_operation(op);
  if (quantization != AffineQuantization::None) {
    validate_affine_quantization(quantization, output, inputs, params);
    const BackendOps& implementation = find_backend(output->backend);
    if (!implementation.dispatch) throw std::runtime_error("Affine quantization backend is unavailable");
    implementation.dispatch(op, output, inputs, params);
    ++counts[backend_index(output->backend)];
    return;
  }
  if (execute_packed_integer(op, output, inputs, params)) return;
  if (packed_integer_cast_operation(op, params)) {
    validate_packed_integer_cast(output, inputs, params);
    const BackendOps& implementation = find_backend(output->backend);
    if (!implementation.dispatch) throw std::runtime_error("Packed integer backend is unavailable");
    implementation.dispatch(op, output, inputs, params);
    ++counts[backend_index(output->backend)];
    return;
  }
  const LowPrecisionAttention low_attention = low_precision_attention_operation(op, params);
  const LowPrecisionRandom low_random = low_precision_random_operation(op, params);
  const LowPrecisionFused low_fused = low_precision_fused_operation(op, params);
  const LowPrecisionLoss low_loss = low_precision_loss_operation(op, params);
  const LowPrecisionSpatial low_spatial = low_precision_spatial_operation(op, params);
  const LowPrecisionIndexing low_indexing = low_precision_indexing_operation(op, params);
  const LowPrecisionNormalization low_normalization = low_precision_normalization_operation(op, params);
  const LowPrecisionSoftmax low_softmax = low_precision_softmax_operation(op, params);
  const LowPrecisionMatrix low_matrix = low_precision_matrix_operation(op, params);
  const LowPrecisionReduction low_reduction = low_precision_reduction_operation(op, params);
  const bool low_precision_cast = low_precision_cast_operation(op, params);
  const LowPrecisionPointwise low_operation = low_precision_pointwise_operation(op, params);
  const Float64Pointwise double_operation = low_operation == LowPrecisionPointwise::None ? float64_pointwise_operation(op, params) : Float64Pointwise::None;
  const Float64Reduction double_reduction = float64_reduction_operation(op, params);
  const Float64Matrix double_matrix = float64_matrix_operation(op, params);
  const Float64Softmax double_softmax = float64_softmax_operation(op, params);
  const Float64Indexing double_indexing = float64_indexing_operation(op, params);
  const Float64Spatial double_spatial = float64_spatial_operation(op, params);
  const Float64Loss double_loss = float64_loss_operation(op, params);
  const Float64Attention double_attention = float64_attention_operation(op, params);
  const Float64Random double_random = float64_random_operation(op, params);
  const bool double_categorical = float64_categorical_operation(op, params);
  const bool low_categorical = low_precision_categorical_operation(op, params);
  const bool low_optimizer = op == "optimizer_step";
  const Float64Normalization double_normalization = float64_normalization_operation(op, params);
  const IntegerPointwise integer_operation = integer_pointwise_operation(op, params);
  const IntegerMatrix integer_matrix = integer_matrix_operation(op, params);
  const StorageLayout storage_layout = storage_layout_operation(op, output->dtype);
  const bool integer_extrema = (op == "max" || op == "min" || op == "amax" || op == "amin") &&
                               is_standard_integer(output->dtype);
  LocalFunction require_kernel_dtype{[&](DType dtype) {
    if (low_random != LowPrecisionRandom::None || low_attention != LowPrecisionAttention::None || low_fused != LowPrecisionFused::None || low_loss != LowPrecisionLoss::None || low_spatial != LowPrecisionSpatial::None || low_indexing != LowPrecisionIndexing::None || (op == "argsort" && is_low_precision_float(dtype)) || low_normalization != LowPrecisionNormalization::None || low_precision_cast || low_operation != LowPrecisionPointwise::None || low_reduction != LowPrecisionReduction::None || low_matrix != LowPrecisionMatrix::None || low_softmax != LowPrecisionSoftmax::None) return;
    if (storage_layout != StorageLayout::None) return;
    if (integer_operation != IntegerPointwise::None) return;
    if (double_operation != Float64Pointwise::None) return;
    if (double_reduction != Float64Reduction::None) return;
    if (double_matrix != Float64Matrix::None) return;
    if (double_softmax != Float64Softmax::None) return;
    if (double_indexing != Float64Indexing::None) return;
    if (double_spatial != Float64Spatial::None) return;
    if (double_loss != Float64Loss::None) return;
    if (double_attention != Float64Attention::None) return;
    if (double_random != Float64Random::None) return;
    if (double_categorical || low_categorical || low_optimizer) return;
    if (double_normalization != Float64Normalization::None) return;
    if (integer_matrix != IntegerMatrix::None) return;
    // Exact integer reductions use Int64 outputs. Index-copy uses Int64 indices.
    // Operator-specific validation below enforces each supported position.
    if ((op == "cumsum" || ((op == "sum" || op == "prod") && output->dtype == DType::Int64)) &&
        is_standard_integer(dtype)) return;
    if (integer_extrema && is_standard_integer(dtype)) return;
    if (op == "argsort" && (is_standard_integer(dtype) || dtype == DType::Float64)) return;
    if (op == "index_copy" && dtype == DType::Int64) return;
    if (!is_compute_float(dtype) && dtype != DType::Int32 && dtype != DType::Bool)
      throw std::invalid_argument(op + ": arithmetic kernel is not implemented for " + dtype_name(dtype));
  }};
  require_kernel_dtype(output->dtype);
  if (params.output_dtype() != int(output->dtype)) throw std::invalid_argument("Output dtype descriptor mismatch");
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]) {
      if (inputs[i]->count > size_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument(op + ": input element count exceeds Int32 indexing");
      require_kernel_dtype(inputs[i]->dtype);
      if (i >= 4 || params.input_dtype(static_cast<KernelInput>(i)) != int(inputs[i]->dtype))
        throw std::invalid_argument("Input dtype descriptor mismatch");
    }
  }
  for (const BufferPtr& input : inputs) {
    if (input && input->backend != output->backend) {
      throw std::invalid_argument("Mixed-backend operation");
    }
  }
  if (low_attention != LowPrecisionAttention::None) validate_low_precision_attention(low_attention, output, inputs, params);
  if (op == "flux_rotary" || op == "flux_modulate" || op == "flux_attention")
    flux2::detail::validate_kernel(op, output, inputs, params);
  if (low_random != LowPrecisionRandom::None) validate_low_precision_random(low_random, output, inputs, params);
  if (op == "dropout" && is_compute_float(output->dtype)) validate_floating_dropout(output, inputs, params);
  if ((op == "uniform" || op == "normal") && is_compute_float(output->dtype)) validate_floating_random(op == "normal", output, inputs, params);
  if (low_fused != LowPrecisionFused::None) validate_low_precision_fused(low_fused, output, inputs, params);
  if (low_loss != LowPrecisionLoss::None) validate_low_precision_loss(low_loss, output, inputs, params);
  if (low_spatial != LowPrecisionSpatial::None) validate_low_precision_spatial(low_spatial, output, inputs, params);
  if (low_indexing != LowPrecisionIndexing::None) validate_low_precision_indexing(low_indexing, output, inputs, params);
  if (low_normalization != LowPrecisionNormalization::None) validate_low_precision_normalization(low_normalization, output, inputs, params);
  if (low_softmax != LowPrecisionSoftmax::None) validate_low_precision_softmax(output, inputs, params);
  if (low_matrix != LowPrecisionMatrix::None) validate_low_precision_matrix(low_matrix, output, inputs, params);
  if (low_reduction != LowPrecisionReduction::None) validate_low_precision_reduction(low_reduction, output, inputs, params);
  if (low_precision_cast) validate_low_precision_cast(output, inputs, params);
  if (low_operation != LowPrecisionPointwise::None) validate_low_precision_pointwise(low_operation, output, inputs, params);
  if (double_softmax != Float64Softmax::None)
    validate_float64_softmax(output, inputs, params);
  if (double_indexing != Float64Indexing::None)
    validate_float64_indexing(double_indexing, output, inputs, params);
  if (double_spatial != Float64Spatial::None)
    validate_float64_spatial(double_spatial, output, inputs, params);
  if (double_loss != Float64Loss::None)
    validate_float64_loss(double_loss, output, inputs, params);
  if (double_attention != Float64Attention::None)
    validate_float64_attention(double_attention, output, inputs, params);
  if (double_random != Float64Random::None)
    validate_float64_random(double_random, output, inputs, params);
  if (double_normalization != Float64Normalization::None)
    validate_float64_normalization(double_normalization, output, inputs, params);
  if (double_categorical) validate_float64_categorical(output, inputs, params);
  if (low_categorical) validate_low_precision_categorical(output, inputs, params);
  if (low_optimizer) validate_low_precision_optimizer(output, inputs, params);
  if (double_matrix != Float64Matrix::None)
    validate_float64_matrix(double_matrix, output, inputs, params);
  if (double_reduction != Float64Reduction::None)
    validate_float64_reduction(double_reduction, output, inputs, params);
  if (double_operation != Float64Pointwise::None)
    validate_float64_pointwise(double_operation, output, inputs, params);
  if (integer_operation != IntegerPointwise::None)
    validate_integer_pointwise(integer_operation, output, inputs, params);
  if (integer_matrix != IntegerMatrix::None)
    validate_integer_matrix(integer_matrix, output, inputs, params);
  if (storage_layout != StorageLayout::None)
    validate_storage_layout(storage_layout, output, inputs, params);
  if (op == "argsort") {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(!inputs.empty() && inputs.size() <= 4 && inputs[0], "Argsort requires one input");
    for (size_t i = 1; i < inputs.size(); ++i) require(!inputs[i], "Unexpected argsort operand");
    require(output->dtype == DType::Int32, "Argsort requires Int32 indices");
    const int rank = params.input_rank(KernelInput::A), axis = params.attribute(0);
    require(rank >= 1 && rank <= 8 && params.output_rank() == rank, "Invalid argsort rank");
    require(axis >= 0 && axis < rank, "Invalid argsort axis");
    require(params.attribute(1) == 0 || params.attribute(1) == 1, "Invalid argsort direction");
    size_t forward = 1, backward = 1;
    for (int d = 0; d < rank; ++d) {
      const int length = params.input_shape(KernelInput::A, d);
      const int reverse = params.input_shape(KernelInput::A, rank - d - 1);
      require(length >= 0 && reverse >= 0 && params.output_shape(d) == length, "Invalid argsort dimensions");
      require((length == 0 || forward <= size_t(INT32_MAX) / size_t(length)) &&
              (reverse == 0 || backward <= size_t(INT32_MAX) / size_t(reverse)), "Argsort dimensions exceed Int32 indexing");
      require(params.input_stride(KernelInput::A, rank - d - 1) == int(backward), "Invalid argsort contiguous stride");
      forward *= size_t(length); backward *= size_t(reverse);
    }
    require(forward == inputs[0]->count && forward == output->count && params.output_count() == int(forward),
            "Argsort shape differs from storage");
    if (output->bytes && inputs[0]->bytes) {
      const uintptr_t a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
      require(a <= b ? b - a >= output->bytes : a - b >= inputs[0]->bytes, "Argsort output overlaps its input");
    }
  }
  if (((op == "sum" || op == "prod" || op == "cumsum") && output->dtype == DType::Int64) || integer_extrema) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(!inputs.empty() && inputs.size() <= 4 && inputs[0] && is_standard_integer(inputs[0]->dtype),
            "Integer reduction requires one integer input");
    for (size_t i = 1; i < inputs.size(); ++i) require(!inputs[i], "Unexpected integer reduction operand");
    require(!integer_extrema || inputs[0]->dtype == output->dtype, "Integer extrema must preserve the input dtype");
    int rank = params.input_rank(KernelInput::A);
    require(rank >= 0 && rank <= 8, "Invalid integer reduction input rank");
    size_t count = 1;
    for (int axis = rank - 1; axis >= 0; --axis) {
      int dimension = params.input_shape(KernelInput::A, axis);
      require(dimension >= 0 && count <= size_t(std::numeric_limits<int>::max()) &&
              params.input_stride(KernelInput::A, axis) == int(count), "Invalid integer reduction contiguous descriptor");
      count = checked_memory_mul(count, size_t(dimension));
    }
    require(count == inputs[0]->count, "Integer reduction shape differs from storage");
    size_t forward_count = 1;
    for (int dimension = 0; dimension < rank; ++dimension) {
      int length = params.input_shape(KernelInput::A, dimension);
      require(length == 0 || forward_count <= size_t(std::numeric_limits<int>::max()) / size_t(length),
              "Integer reduction prefix dimensions exceed Int32 indexing");
      forward_count *= size_t(length);
    }
    bool scan = op == "cumsum";
    int axis = params.attribute(0), keep = scan ? 1 : params.attribute(1);
    require(keep == 0 || keep == 1, "Invalid integer reduction keepdim flag");
    require(scan ? (rank ? axis >= 0 && axis < rank : axis == 0) : axis >= -1 && axis < rank,
            "Invalid integer reduction axis");
    require(!integer_extrema || (axis < 0 ? count > 0 : params.input_shape(KernelInput::A, axis) > 0),
            "Integer extrema require a nonempty reduction axis");
    std::vector<int> shape;
    for (int dimension = 0; dimension < rank; ++dimension) {
      bool reduced = !scan && (axis < 0 || axis == dimension);
      if (!reduced || keep) shape.push_back(reduced ? 1 : params.input_shape(KernelInput::A, dimension));
    }
    require(params.output_rank() == int(shape.size()), "Invalid integer reduction output rank");
    size_t output_count = 1;
    for (size_t dimension = 0; dimension < shape.size(); ++dimension) {
      require(params.output_shape(int(dimension)) == shape[dimension], "Invalid integer reduction output dimensions");
      output_count = checked_memory_mul(output_count, size_t(shape[dimension]));
    }
    require(output_count == output->count && params.output_count() == int(output_count),
            "Integer reduction output storage differs from shape");
    if (output->bytes && inputs[0]->bytes) {
      uintptr_t first = reinterpret_cast<uintptr_t>(output->raw_pointer());
      uintptr_t second = reinterpret_cast<uintptr_t>(inputs[0]->raw_pointer());
      bool overlap = first <= second ? second - first < output->bytes : first - second < inputs[0]->bytes;
      require(!overlap, "Integer reduction output overlaps its input");
    }
  }
  if (op == "group_norm_weight_grad" && double_normalization == Float64Normalization::None && low_normalization == LowPrecisionNormalization::None) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(inputs.size() >= 2 && inputs.size() <= 4 && inputs[0] && inputs[1], "Group norm weight gradient requires values and gradients");
    for (size_t i = 2; i < inputs.size(); ++i) require(!inputs[i], "Unexpected group norm weight gradient operand");
    require(is_compute_float(inputs[0]->dtype) && inputs[1]->dtype == inputs[0]->dtype &&
            (output->dtype == inputs[0]->dtype || (output->backend == Backend::CPU && output->dtype == DType::Float32)),
            "Invalid group norm weight gradient dtype");
    int rank = params.input_rank(KernelInput::A);
    require(rank >= 2 && rank <= 8 && params.input_rank(KernelInput::B) == rank, "Invalid group norm weight gradient rank");
    size_t count = 1;
    for (int axis = rank - 1; axis >= 0; --axis) {
      int dimension = params.input_shape(KernelInput::A, axis);
      require(dimension >= 0 && count <= size_t(std::numeric_limits<int>::max()) &&
              params.input_shape(KernelInput::B, axis) == dimension &&
              params.input_stride(KernelInput::A, axis) == int(count) && params.input_stride(KernelInput::B, axis) == int(count),
              "Invalid contiguous group norm weight gradient descriptor");
      count = checked_memory_mul(count, size_t(dimension));
    }
    require(count == inputs[0]->count && count == inputs[1]->count, "Group norm weight gradient storage differs from shape");
    int channels = params.input_shape(KernelInput::A, 1), groups = params.attribute(0);
    require(channels > 0 && groups > 0 && channels % groups == 0 &&
            (params.attribute(2) == 0 || params.attribute(2) == 1) && group_norm_epsilon(params) > 0,
            "Invalid group norm weight gradient parameters");
    require(params.output_rank() == 1 && params.output_shape(0) == channels && params.output_count() == channels &&
            output->count == size_t(channels), "Invalid group norm weight gradient output");
    for (size_t i = 0; i < 2; ++i) if (output->bytes && inputs[i]->bytes) {
      uintptr_t first = reinterpret_cast<uintptr_t>(output->raw_pointer()), second = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
      bool overlap = first <= second ? second - first < output->bytes : first - second < inputs[i]->bytes;
      require(!overlap, "Group norm weight gradient output overlaps an input");
    }
  }
  if (op == "index_copy") {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require((storage_layout == StorageLayout::IndexCopy || is_compute_float(output->dtype) || output->dtype == DType::Int32 || output->dtype == DType::Bool) &&
            inputs.size() >= 3 && inputs.size() <= 4 && inputs[0] && inputs[1] && inputs[2] && (inputs.size() == 3 || !inputs[3]),
            "Index copy requires destination, indices and source");
    require(inputs[0]->dtype == output->dtype && inputs[2]->dtype == output->dtype &&
            (inputs[1]->dtype == DType::Int32 || inputs[1]->dtype == DType::Int64), "Invalid index copy dtype");
    KernelParams p = params;
    for (int i = 0; i < 3; ++i) {
      KernelInput input = static_cast<KernelInput>(i);
      int rank = p.input_rank(input);
      require(rank >= 0 && rank <= 8 && (i != 1 || rank <= 1), "Invalid index copy rank");
      size_t count = 1;
      for (int axis = rank - 1; axis >= 0; --axis) {
        int dimension = p.input_shape(input, axis);
        require(dimension >= 0 && count <= size_t(std::numeric_limits<int>::max()) &&
                p.input_stride(input, axis) == int(count), "Invalid contiguous index copy descriptor");
        count = checked_memory_mul(count, size_t(dimension));
      }
      require(count == inputs[size_t(i)]->count, "Index copy descriptor differs from storage");
    }
    int rank = p.input_rank(KernelInput::A);
    require(p.output_rank() == rank && output->count == inputs[0]->count &&
            p.output_count() == int(output->count), "Index copy output shape differs");
    for (int axis = 0; axis < rank; ++axis)
      require(p.output_shape(axis) == p.input_shape(KernelInput::A, axis), "Index copy output dimensions differ");
    // Treat scalar operands as having a virtual length-one axis.
    for (KernelInput input : {KernelInput::A, KernelInput::B, KernelInput::C}) if (!p.input_rank(input)) {
      int slot = KernelParams::input_slot(input);
      p.v[slot] = 1;
      p.v[slot + KernelParams::kInputShapeOffset] = 1;
      p.v[slot + KernelParams::kInputStrideOffset] = 1;
    }
    if (!rank) {
      rank = 1;
      p.v[KernelParams::kOutputRankSlot] = 1;
      p.v[KernelParams::kOutputShapeSlot] = 1;
    }
    int axis = p.attribute(operator_attr::index_copy::kAxis);
    if (axis < 0) axis += rank;
    require(axis >= 0 && axis < rank && p.input_rank(KernelInput::C) == rank, "Invalid index copy axis/source rank");
    require(size_t(p.input_shape(KernelInput::C, axis)) == inputs[1]->count, "Index copy source/index length differs");
    for (int dimension = 0; dimension < rank; ++dimension)
      if (dimension != axis) require(p.input_shape(KernelInput::A, dimension) == p.input_shape(KernelInput::C, dimension),
                                    "Index copy slice shapes differ");
    bool (*const overlaps)(const BufferPtr&, const BufferPtr&) = [](const BufferPtr& a, const BufferPtr& b) {
      if (!a->bytes || !b->bytes) return false;
      uintptr_t first = reinterpret_cast<uintptr_t>(a->raw_pointer()), second = reinterpret_cast<uintptr_t>(b->raw_pointer());
      return first <= second ? second - first < a->bytes : first - second < b->bytes;
    };
    require(!overlaps(output, inputs[0]) || output->raw_pointer() == inputs[0]->raw_pointer(),
            "Index copy destination partially overlaps its base");
    require(!overlaps(output, inputs[1]) && !overlaps(output, inputs[2]), "Index copy source/indices overlap destination");
    bool duplicate = false;
    // Empty slices bypass index-value checks.
    if (inputs[2]->count) {
      MemoryReservation host(Backend::CPU, checked_memory_add(inputs[1]->bytes, checked_memory_mul(inputs[1]->count, sizeof(int32_t))));
      std::vector<uint8_t> bytes = backend_download_raw(inputs[1]);
      std::vector<int32_t> values(inputs[1]->count);
      for (size_t i = 0; i < values.size(); ++i) {
        int64_t value;
        if (inputs[1]->dtype == DType::Int64) std::memcpy(&value, bytes.data() + i * 8, 8);
        else { int32_t narrow; std::memcpy(&narrow, bytes.data() + i * 4, 4); value = narrow; }
        require(value >= 0 && value < p.input_shape(KernelInput::A, axis), "Index copy index out of range");
        values[i] = int32_t(value);
      }
      std::sort(values.begin(), values.end());
      duplicate = std::adjacent_find(values.begin(), values.end()) != values.end();
    }
    p.v[KernelParams::kAttributeSlot + operator_attr::index_copy::kAxis] = axis;
    p.v[KernelParams::kAttributeSlot + operator_attr::index_copy::kHasDuplicates] = int(duplicate);
    const BackendOps& implementation = find_backend(output->backend);
    if (!implementation.dispatch) throw std::runtime_error("Index copy backend not compiled");
    implementation.dispatch(op, output, inputs, p);
    ++counts[backend_index(output->backend)];
    return;
  }
  if (op == "packed_linear" && low_matrix == LowPrecisionMatrix::None) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require((is_arithmetic_float(output->dtype) || (is_standard_integer(output->dtype) && output->dtype != DType::Bool)) && params.output_rank() == 1 &&
            inputs.size() >= 2 && inputs[0] && inputs[1], "Packed linear requires numeric input and weight");
    require(inputs.size() < 4 || !inputs[3], "Packed linear accepts at most three defined inputs");
    for (int i = 0; i < 3; ++i) {
      KernelInput input = static_cast<KernelInput>(i);
      bool present = size_t(i) < inputs.size() && bool(inputs[i]);
      int rank = params.input_rank(input);
      if (!present) { require(i == 2 && rank == -1, "Missing packed linear input descriptor"); continue; }
      require(inputs[i]->dtype == output->dtype && rank >= 1 && rank <= 8 && (i == 0 || rank == (i == 1 ? 2 : 1)),
              "Invalid packed linear rank or dtype");
      size_t count = 1;
      for (int axis = rank - 1; axis >= 0; --axis) {
        int dimension = params.input_shape(input, axis);
        require(dimension >= 0 && count <= size_t(std::numeric_limits<int>::max()) && params.input_stride(input, axis) == int(count),
                "Packed linear requires valid contiguous descriptors");
        count = checked_memory_mul(count, size_t(dimension));
      }
      require(count == inputs[i]->count, "Packed linear input shape differs from storage");
    }
    int rank = params.input_rank(KernelInput::A), columns = params.input_shape(KernelInput::B, 0);
    require(params.input_shape(KernelInput::A, rank - 1) == params.input_shape(KernelInput::B, 1),
            "Packed linear input feature dimensions differ");
    require(params.input_rank(KernelInput::C) < 0 || params.input_shape(KernelInput::C, 0) == columns,
            "Packed linear bias shape differs");
    int sections = params.attribute(operator_attr::packed_linear::kCount), total = 0;
    require(sections > 0 && sections < KernelParams::kAttributeCount, "Invalid packed projection count");
    for (int i = 0; i < sections; ++i) {
      int width = params.attribute(operator_attr::packed_linear::kWidths + i);
      require(width >= 0 && width <= columns - total, "Invalid packed projection width");
      total += width;
    }
    require(total == columns, "Packed projection widths differ from weight shape");
    size_t rows = 1;
    for (int axis = 0; axis + 1 < rank; ++axis) {
      rows = checked_memory_mul(rows, size_t(params.input_shape(KernelInput::A, axis)));
      require(rows <= size_t(std::numeric_limits<int>::max()), "Packed linear row count exceeds Int32");
    }
    size_t count = checked_memory_mul(rows, size_t(columns));
    require(count == output->count && count <= size_t(std::numeric_limits<int>::max()) &&
            params.output_count() == int(count) && params.output_shape(0) == int(count), "Invalid packed linear output shape");
  }
  if (op == "tiled_attention" && low_attention == LowPrecisionAttention::None) {
    void (*const require)(bool, const char*) = [](bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); };
    require(is_arithmetic_float(output->dtype) && params.output_rank() == 4 && inputs.size() == 4,
            "Tiled attention requires four floating inputs and a rank-four output");
    for (int i = 0; i < 4; ++i) {
      KernelInput input = static_cast<KernelInput>(i);
      int rank = params.input_rank(input);
      require(inputs[i] && inputs[i]->dtype == output->dtype && rank >= 0 && rank <= 4 && (i == 3 || rank == 4),
              "Invalid tiled attention input rank or dtype");
      size_t count = 1;
      for (int d = rank - 1; d >= 0; --d) {
        int dimension = params.input_shape(input, d);
        require(dimension >= 0 && count <= size_t(std::numeric_limits<int>::max()) && params.input_stride(input, d) == int(count),
                "Tiled attention requires valid contiguous input descriptors");
        count = checked_memory_mul(count, size_t(dimension));
      }
      require(count == inputs[i]->count, "Tiled attention input shape differs from its storage");
    }
    LocalFunction q{[&](int d) { return params.input_shape(KernelInput::A, d); }};
    LocalFunction k{[&](int d) { return params.input_shape(KernelInput::B, d); }};
    LocalFunction v{[&](int d) { return params.input_shape(KernelInput::C, d); }};
    require(q(1) > 0 && k(1) > 0 && q(1) % k(1) == 0 && k(1) == v(1) &&
            q(0) == k(0) && q(0) == v(0) && q(3) == k(3) && k(2) == v(2),
            "Incompatible tiled attention batch, head or feature dimensions");
    int expected[4] = {q(0), q(1), q(2), v(3)}, scores[4] = {q(0), q(1), q(2), k(2)};
    size_t output_count = 1;
    for (int d = 0; d < 4; ++d) {
      require(params.output_shape(d) == expected[d], "Tiled attention output shape differs");
      output_count = checked_memory_mul(output_count, size_t(expected[d]));
    }
    require(output_count == output->count && output_count == size_t(params.output_count()), "Invalid tiled attention output count");
    int rank = params.input_rank(KernelInput::D);
    for (int d = 0; d < rank; ++d)
      require(params.input_shape(KernelInput::D, d) == 1 || params.input_shape(KernelInput::D, d) == scores[4 - rank + d],
              "Tiled attention mask does not broadcast to the scores");
    int tile = params.attribute(operator_attr::tiled_attention::kQueryTile);
    require(tile > 0, "Tiled attention query tile must be positive");
    int causal = params.attribute(operator_attr::tiled_attention::kCausal);
    require(causal == 0 || causal == 1, "Invalid tiled attention causal flag");
    int tiling = params.attribute(operator_attr::tiled_attention::kTiling);
    require(tiling == 0 || tiling == 1, "Invalid attention tiling mode");
    size_t tile_elements = tiling ? checked_memory_mul(size_t(q(2)), size_t(k(2))) : size_t(q(0));
    if (!tiling) for (int value : {q(1), tile < q(2) ? tile : q(2), k(2)}) tile_elements = checked_memory_mul(tile_elements, size_t(value));
    require(tile_elements <= size_t(std::numeric_limits<int>::max()), "Tiled attention workspace exceeds the Int32 element limit");
  }
  if (op == "attention_softmax" && low_attention == LowPrecisionAttention::None) {
    if (params.attribute(operator_attr::attention_softmax::kCausal) ||
        params.attribute(operator_attr::attention_softmax::kCausalOffset) ||
        params.attribute(operator_attr::attention_softmax::kQueryBase))
      throw std::invalid_argument("Causal softmax attributes are internal to tiled attention");
    int rank = params.output_rank(), axis = params.attribute(operator_attr::attention_softmax::kAxis);
    if (rank < 0 || rank > 8 || axis < 0 || axis >= (rank ? rank : 1) ||
        !is_arithmetic_float(output->dtype) || inputs.size() < 2 || !inputs[0] || !inputs[1] ||
        inputs[0]->dtype != output->dtype || inputs[1]->dtype != output->dtype ||
        params.input_rank(KernelInput::A) != rank)
      throw std::invalid_argument("Invalid attention softmax dtype, rank or axis");
    for (int d = 0; d < rank; ++d)
      if (params.input_shape(KernelInput::A, d) != params.output_shape(d))
        throw std::invalid_argument("Attention softmax output shape must equal its scores");
    int mask_rank = params.input_rank(KernelInput::B);
    if (mask_rank < 0 || mask_rank > rank) throw std::invalid_argument("Invalid attention mask rank");
    for (int d = 0; d < mask_rank; ++d) {
      int size = params.input_shape(KernelInput::B, d);
      if (size != 1 && size != params.output_shape(rank - mask_rank + d))
        throw std::invalid_argument("Attention mask does not broadcast to the scores");
    }
  }
  if (op == "rms_norm" && double_normalization == Float64Normalization::None && low_normalization == LowPrecisionNormalization::None) {
    int rank = params.output_rank(), trailing = params.attribute(operator_attr::normalization::kTrailingDimensions);
    int rounding = params.attribute(operator_attr::normalization::kRounding);
    float epsilon;
    int bits = params.attribute(operator_attr::normalization::kEpsilon);
    std::memcpy(&epsilon, &bits, sizeof(epsilon));
    if (rank < 1 || rank > 8 || trailing < 1 || trailing > rank || rounding < 0 || rounding > 1 ||
        !(epsilon >= 0) || !is_compute_float(output->dtype) || inputs.empty() || !inputs[0] ||
        inputs[0]->dtype != output->dtype || params.input_rank(KernelInput::A) != rank)
      throw std::invalid_argument("Invalid RMS normalization contract");
    for (int axis = 0; axis < rank; ++axis)
      if (params.input_shape(KernelInput::A, axis) != params.output_shape(axis))
        throw std::invalid_argument("RMS normalization output shape must equal its input");
    for (int k = 1; k < int(inputs.size()); ++k) {
      if (!inputs[k]) continue;
      KernelInput source = static_cast<KernelInput>(k);
      if (!is_compute_float(inputs[k]->dtype) || (rounding && inputs[k]->dtype != output->dtype) ||
          params.input_rank(source) != trailing)
        throw std::invalid_argument("RMS affine inputs require floating dtype and normalized shape; rounding mode 1 requires matching dtypes");
      for (int axis = 0; axis < trailing; ++axis)
        if (params.input_shape(source, axis) != params.output_shape(rank - trailing + axis))
          throw std::invalid_argument("RMS affine shape differs from the normalized dimensions");
    }
  }
  if ((op == "silu_mul" || op == "rotary_embedding") && low_fused == LowPrecisionFused::None) {
    const int arity = op == "silu_mul" ? 2 : 3, rank = params.output_rank();
    if (!is_arithmetic_float(output->dtype) || rank < 0 || rank > 8 || inputs.size() < size_t(arity))
      throw std::invalid_argument("Invalid fused pointwise output or inputs");
    for (int k = 0; k < arity; ++k) {
      KernelInput source = static_cast<KernelInput>(k);
      int input_rank = params.input_rank(source);
      if (!inputs[k] || inputs[k]->dtype != output->dtype || input_rank < 0 || input_rank > rank)
        throw std::invalid_argument("Fused pointwise inputs require matching floating dtypes and broadcastable ranks");
      for (int axis = 0; axis < input_rank; ++axis) {
        int size = params.input_shape(source, axis);
        if (size != 1 && size != params.output_shape(rank - input_rank + axis))
          throw std::invalid_argument("Fused pointwise input does not broadcast to the output");
      }
    }
    if (op == "rotary_embedding") {
      if (rank < 1 || params.output_shape(rank - 1) % 2 || params.input_rank(KernelInput::A) != rank)
        throw std::invalid_argument("Rotary input requires an even trailing dimension");
      for (int axis = 0; axis < rank; ++axis)
        if (params.input_shape(KernelInput::A, axis) != params.output_shape(axis))
          throw std::invalid_argument("Rotary output shape must equal its input");
    }
  }
  if ((op == "cumsum" || op == "cumsum_backward") && low_reduction == LowPrecisionReduction::None) {
    if (inputs.empty() || !inputs[0] || inputs[0]->count != output->count)
      throw std::invalid_argument("Cumsum requires equal input/output element counts");
    DType source = inputs[0]->dtype;
    DType expected = is_arithmetic_float(source) ? source : DType::Int64;
    if (output->dtype != expected || (op == "cumsum_backward" && !is_arithmetic_float(source)))
      throw std::invalid_argument("Invalid cumsum output dtype");
  }
  if (op == "token_presence") {
    if (inputs.empty() || !inputs[0] || inputs[0]->dtype != DType::Int32 || output->dtype != DType::Int32 ||
        params.input_rank(KernelInput::A) != 2 || params.output_rank() != 2 || params.output_shape(1) < 2 ||
        params.input_shape(KernelInput::A, 0) != params.output_shape(0))
      throw std::invalid_argument("Invalid token_presence shape or dtype");
  }
  if (op == "categorical" && !double_categorical && !low_categorical) {
    if (inputs.size() < 3 || !inputs[0] || !inputs[1] || !inputs[2] ||
        !is_compute_float(inputs[0]->dtype) || inputs[1]->dtype != inputs[0]->dtype ||
        !is_compute_float(inputs[2]->dtype) || output->dtype != DType::Int32 ||
        params.output_rank() != 1 || params.input_rank(KernelInput::A) != 2 ||
        params.input_rank(KernelInput::B) != 2 || params.input_rank(KernelInput::C) != 1 ||
        params.input_shape(KernelInput::A, 1) <= 0 ||
        params.input_shape(KernelInput::A, 0) != params.output_count() ||
        params.input_shape(KernelInput::B, 0) != params.output_count() ||
        params.input_shape(KernelInput::B, 1) != params.input_shape(KernelInput::A, 1) ||
        inputs[2]->count != output->count)
      throw std::invalid_argument("Invalid categorical shape or dtype");
  }
  if (low_attention == LowPrecisionAttention::Softmax || low_attention == LowPrecisionAttention::Tiled) {
    execute_low_precision_attention(low_attention, output, inputs, params);
    ++counts[backend_index(output->backend)];
    return;
  }
  const BackendOps& implementation = find_backend(output->backend);
  if (!implementation.dispatch) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  implementation.dispatch(op, output, inputs, params);
  ++counts[backend_index(output->backend)];
}

uint64_t backend_dispatch_count(Backend backend) {
  return counts[backend_index(backend)].load();
}
}  // namespace elman
