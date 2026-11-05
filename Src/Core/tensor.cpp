#include <algorithm>
#include <limits>
#include <stdexcept>

#include "Core/Graph.h"
#include "Core/TensorDispatch.h"

namespace elman {
static void require(bool ok, const std::string& message) {
  if (!ok) {
    throw std::invalid_argument(message);
  }
}
size_t shape_element_count(const Shape& shape) {
  require(shape.size() <= 8, "Maximum tensor rank is eight");
  size_t n = 1;
  for (int d : shape) {
    require(d > 0, "Tensor dimensions must be positive");
    require(n <= size_t(std::numeric_limits<int>::max()) / size_t(d),
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
DType Tensor::dtype() const {
  require_defined(*this);
  return node_->value->dtype;
}
std::vector<float> Tensor::data() const {
  require_defined(*this);
  return backend_download(node_->value);
}
void Tensor::assign(const std::vector<float>& values) {
  require_defined(*this);
  backend_upload(node_->value, values);
  ++node_->value->version;
}
Tensor Tensor::from_data(Shape shape, const std::vector<float>& data, Backend b,
                         bool grad, DType dtype) {
  require(shape_element_count(shape) == data.size(),
          "Data and tensor shape sizes differ");
  require(!grad || dtype == DType::Float32,
          "Only float32 tensors can require gradients");
  auto n = std::make_shared<TensorNode>();
  n->shape = std::move(shape);
  n->value = backend_allocate(b, data.size(), dtype);
  n->requires_grad = grad;
  backend_upload(n->value, data);
  return Tensor(n);
}
Tensor Tensor::zeros(Shape s, Backend b, bool grad) {
  return from_data(s, std::vector<float>(shape_element_count(s), 0), b, grad);
}
Tensor Tensor::ones(Shape s, Backend b, bool grad) {
  return from_data(s, std::vector<float>(shape_element_count(s), 1), b, grad);
}
Tensor Tensor::to(Backend b) const {
  return from_data(shape(), data(), b, requires_grad(), dtype());
}
Tensor dispatch_tensor(const std::string& op, const Shape& shape,
                       const std::vector<Tensor>& inputs,
                       const std::vector<int>& attrs, DType dtype,
                       Backend backend) {
  auto n = std::make_shared<TensorNode>();
  n->shape = shape;
  if (!inputs.empty()) {
    backend = inputs[0].backend();
  }
  n->value = backend_allocate(backend, shape_element_count(shape), dtype);
  n->operation = op;
  KernelParams p;
  p.v[KernelParams::kOutputCountSlot] = int(shape_element_count(shape));
  p.v[KernelParams::kOutputRankSlot] = int(shape.size());
  std::copy(shape.begin(), shape.end(), p.v + KernelParams::kOutputShapeSlot);
  require(inputs.size() <= 3, "Kernel accepts at most three inputs");
  std::vector<BufferPtr> buffers;
  for (int k = 0; k < 3; ++k) {
    int base = KernelParams::kInputRankSlot + KernelParams::kInputSlotWidth * k;
    p.v[base] = -1;
    if (k >= int(inputs.size()) || !inputs[k].defined()) {
      buffers.push_back(nullptr);
      continue;
    }
    const Tensor& t = inputs[k];
    require(t.backend() == backend, "Mixed-backend operation");
    p.v[base] = int(t.shape().size());
    std::copy(t.shape().begin(), t.shape().end(),
              p.v + base + KernelParams::kInputShapeOffset);
    int stride = 1;
    for (int d = int(t.shape().size()) - 1; d >= 0; --d) {
      p.v[base + KernelParams::kInputStrideOffset + d] = stride;
      stride *= t.shape()[d];
    }
    p.v[KernelParams::kInputDTypeSlot + k] = int(t.dtype());
    buffers.push_back(t.node()->value);
  }
  require(attrs.size() <=
              KernelParams::kInputDTypeSlot - KernelParams::kAttributeSlot,
          "Too many kernel attributes");
  std::copy(attrs.begin(), attrs.end(), p.v + KernelParams::kAttributeSlot);
  p.v[KernelParams::kOutputDTypeSlot] = int(dtype);
  backend_dispatch(op, n->value, buffers, p);
  return Tensor(n);
}
}  // namespace elman
