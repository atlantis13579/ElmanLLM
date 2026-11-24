#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

#include "Core/Graph.h"
#include "Core/Autograd.h"
#include "Core/LowPrecisionFloat.h"
#include "Core/TensorDispatch.h"
#include "Core/Device.h"

namespace elman {
static void require(bool ok, const char* message) {
  if (!ok) {
    throw std::invalid_argument(message);
  }
}
size_t shape_element_count(const Shape& shape) {
  require(shape.size() <= 8, "Maximum tensor rank is eight");
  size_t n = 1;
  for (int d : shape) {
    require(d >= 0, "Tensor dimensions must be nonnegative");
    require(d == 0 || n <= size_t(std::numeric_limits<int>::max()) / size_t(d),
            "Tensor exceeds 32-bit indexing limit");
    n *= size_t(d);
  }
  return n;
}
static void require_defined(const Tensor& x) {
  require(x.defined(), "Undefined tensor");
}
const Shape& Tensor::shape() const {
  require_defined(*this);
  return node_->shape;
}
size_t Tensor::numel() const {
  require_defined(*this);
  return node_->value->count;
}
Backend Tensor::backend() const {
  require_defined(*this);
  return node_->value->backend;
}
size_t Tensor::nbytes() const {
  require_defined(*this);
  return node_->value->bytes;
}
std::vector<uint8_t> Tensor::raw_data() const {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  return backend_download_raw(node_->value);
}
void Tensor::assign_raw(const std::vector<uint8_t>& bytes) {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  backend_upload_raw(node_->value, bytes);
  ++node_->value->version;
}
Tensor Tensor::from_raw(Shape shape, DType dtype, const std::vector<uint8_t>& bytes,
                        Backend backend, bool grad) {
  size_t count = shape_element_count(shape);
  require(bytes.size() == dtype_storage_bytes(dtype, count), "Raw data and tensor shape sizes differ");
  require(!grad || is_differentiable_float(dtype) || is_complex(dtype), "Gradients are not implemented for this storage dtype");
  std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>();
  node->shape = std::move(shape);
  node->value = backend_allocate(backend, count, dtype);
  node->requires_grad = grad;
  backend_upload_raw(node->value, bytes);
  return Tensor(node);
}
DType Tensor::dtype() const {
  require_defined(*this);
  return node_->value->dtype;
}
std::vector<float> Tensor::data() const {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  return backend_download(node_->value);
}
void Tensor::assign(const std::vector<float>& values) {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  if (is_complex(dtype())) {
    assign_complex(std::vector<std::complex<double>>(values.begin(), values.end()));
    return;
  }
  backend_upload(node_->value, values);
  ++node_->value->version;
}
std::vector<double> Tensor::double_data() const {
  require(dtype() == DType::Float64, "Double data requires Float64 storage");
  const std::vector<uint8_t> bytes = raw_data();
  std::vector<double> result(numel());
  if (!bytes.empty()) std::memcpy(result.data(), bytes.data(), bytes.size());
  return result;
}
void Tensor::assign_double(const std::vector<double>& values) {
  require(dtype() == DType::Float64 && numel() == values.size(), "Double assignment dtype or shape mismatch");
  std::vector<uint8_t> bytes(values.size() * sizeof(double));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  assign_raw(bytes);
}
Tensor Tensor::from_double_data(Shape shape, const std::vector<double>& values, Backend backend, bool grad) {
  require(shape_element_count(shape) == values.size(), "Double data and tensor shape sizes differ");
  std::vector<uint8_t> bytes(values.size() * sizeof(double));
  if (!bytes.empty()) std::memcpy(bytes.data(), values.data(), bytes.size());
  return from_raw(std::move(shape), DType::Float64, bytes, backend, grad);
}
std::vector<int32_t> Tensor::int_data() const {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  return backend_download_int(node_->value);
}
void Tensor::assign_int(const std::vector<int32_t>& values) {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  backend_upload_int(node_->value, values);
  ++node_->value->version;
}
std::vector<int64_t> Tensor::int64_data() const {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  return backend_download_int64(node_->value);
}
void Tensor::assign_int64(const std::vector<int64_t>& values) {
  require_defined(*this);
  DeviceScope owner(backend(), device());
  backend_upload_int64(node_->value, values);
  ++node_->value->version;
}
Tensor Tensor::from_int64_data(Shape shape, const std::vector<int64_t>& data, Backend backend) {
  require(shape_element_count(shape) == data.size(), "Int64 data and tensor shape sizes differ");
  std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>();
  node->shape = std::move(shape);
  node->value = backend_allocate(backend, data.size(), DType::Int64);
  backend_upload_int64(node->value, data);
  return Tensor(node);
}
Tensor Tensor::from_int_data(Shape shape, const std::vector<int32_t>& data,
                             Backend backend, DType dtype) {
  require(dtype == DType::Int32 || dtype == DType::Bool || is_packed_integer(dtype),
          "Integer data requires int32, bool, or packed integer dtype");
  require(shape_element_count(shape) == data.size(), "Integer data and tensor shape sizes differ");
  std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>();
  node->shape = std::move(shape);
  node->value = backend_allocate(backend, data.size(), dtype);
  backend_upload_int(node->value, data);
  return Tensor(node);
}
Tensor Tensor::from_data(Shape shape, const std::vector<float>& data, Backend b,
                         bool grad, DType dtype) {
  require(shape_element_count(shape) == data.size(),
          "Data and tensor shape sizes differ");
  require(!grad || is_differentiable_float(dtype) || is_complex(dtype),
          "Only implemented floating tensors can require gradients");
  if (is_complex(dtype)) return from_complex_data(shape,
      std::vector<std::complex<double>>(data.begin(), data.end()), b, grad, dtype);
  std::shared_ptr<TensorNode> n = std::make_shared<TensorNode>();
  n->shape = std::move(shape);
  n->value = backend_allocate(b, data.size(), dtype);
  n->requires_grad = grad;
  backend_upload(n->value, data);
  return Tensor(n);
}
Tensor Tensor::zeros(Shape s, Backend b, bool grad, DType dtype) {
  if (b == Backend::CUDA) {
    require(!grad || is_differentiable_float(dtype) || is_complex(dtype),
            "Only implemented floating tensors can require gradients");
    std::shared_ptr<TensorNode> node = std::make_shared<TensorNode>();
    node->shape = std::move(s);
    node->value = backend_allocate(b, shape_element_count(node->shape), dtype);
    node->requires_grad = grad;
    return Tensor(node);
  }
  return from_data(s, std::vector<float>(shape_element_count(s), 0), b, grad, dtype);
}
Tensor Tensor::ones(Shape s, Backend b, bool grad, DType dtype) {
  return from_data(s, std::vector<float>(shape_element_count(s), 1), b, grad, dtype);
}
Tensor Tensor::to(Backend b) const {
  return to(b, current_device(b));
}
int Tensor::device() const {
  require(defined(), "Undefined tensor device");
  return node_->value->device_ordinal();
}
Tensor Tensor::to_device(int ordinal) const { return to(backend(), ordinal); }
Tensor Tensor::to(Backend b, int ordinal) const {
  if (b == backend() && ordinal == device()) return *this;
  const Tensor source = *this;
  Tensor result;
  if (b == Backend::CUDA && backend() == Backend::CUDA) {
    DeviceScope destination(b, ordinal);
    result = Tensor::zeros(shape(), b, false, dtype());
    copy_device_storage(result.node()->value, node()->value);
  } else {
    std::vector<uint8_t> bytes;
    { DeviceScope owner(backend(), device()); bytes = raw_data(); }
    DeviceScope destination(b, ordinal);
    result = from_raw(shape(), dtype(), bytes, b);
  }
  return graph::tracked(result, {source},
                        [source](const Tensor& gradient) {
                          DeviceScope owner(source.backend(), source.device());
                          autograd::accumulate(source, gradient.to(source.backend(), source.device()));
                        });
}
static void dispatch_bound(const std::string& op, const Tensor& output,
                          const std::vector<Tensor>& inputs, const std::vector<int>& attrs,
                          size_t count, int device) {
  const Shape& shape = output.shape();
  Backend backend = output.backend();
  KernelParams p;
  p.v[KernelParams::kOutputCountSlot] = int(count);
  p.v[KernelParams::kOutputRankSlot] = int(shape.size());
  std::copy(shape.begin(), shape.end(), p.v + KernelParams::kOutputShapeSlot);
  require(inputs.size() <= 4, "Kernel accepts at most four inputs");
  std::vector<BufferPtr> buffers;
  buffers.reserve(4);
  for (int k = 0; k < 4; ++k) {
    int base = KernelParams::input_slot(static_cast<KernelInput>(k));
    p.v[base] = -1;
    if (k >= int(inputs.size()) || !inputs[k].defined()) {
      buffers.push_back(nullptr);
      continue;
    }
    const Tensor& t = inputs[k];
    require(t.backend() == backend, "Mixed-backend operation");
    require(t.device() == device, "Mixed-device operation requires an explicit transfer");
    p.v[base] = int(t.shape().size());
    std::copy(t.shape().begin(), t.shape().end(),
              p.v + base + KernelParams::kInputShapeOffset);
    int stride = 1;
    for (int d = int(t.shape().size()) - 1; d >= 0; --d) {
      p.v[base + KernelParams::kInputStrideOffset + d] = stride;
      stride *= t.shape()[d];
    }
    p.v[KernelParams::input_dtype_slot(static_cast<KernelInput>(k))] = int(t.dtype());
    buffers.push_back(t.node()->value);
  }
  require(attrs.size() <= KernelParams::kAttributeCount,
          "Too many kernel attributes");
  std::copy(attrs.begin(), attrs.end(), p.v + KernelParams::kAttributeSlot);
  p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
  backend_dispatch(op, output.node()->value, buffers, p);
}
Tensor dispatch_tensor(const std::string& op, const Shape& shape,
                       const std::vector<Tensor>& inputs,
                       const std::vector<int>& attrs, DType dtype,
                       Backend backend) {
  std::shared_ptr<TensorNode> n = std::make_shared<TensorNode>();
  n->shape = shape;
  if (!inputs.empty()) {
    backend = inputs[0].backend();
  }
  const int device = inputs.empty() ? current_device(backend) : inputs[0].device();
  DeviceScope owner(backend, device);
  const size_t count = shape_element_count(shape);
  n->value = detail::allocate_dispatch_output(op, backend, count, dtype, inputs);
  n->operation = op;
  dispatch_bound(op, Tensor(n), inputs, attrs, count, device);
  return Tensor(n);
}
void dispatch_tensor_into(const std::string& op, const Tensor& output,
                          const std::vector<Tensor>& inputs, const std::vector<int>& attrs) {
  const int device = output.device();
  DeviceScope owner(output.backend(), device);
  dispatch_bound(op, output, inputs, attrs, shape_element_count(output.shape()), device);
}
}  // namespace elman
