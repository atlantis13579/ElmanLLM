#include "Core/Autograd.h"
#include "Core/Device.h"

#include <stdexcept>

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

bool Tensor::requires_grad() const {
  return node_ && node_->requires_grad;
}

std::vector<float> Tensor::grad() const {
  require(defined(), "Undefined tensor");
  return node_->gradient ? Tensor(node_->gradient).data()
                         : std::vector<float>{};
}

std::vector<double> Tensor::double_grad() const {
  require(defined() && dtype() == DType::Float64, "Double gradients require Float64 tensors");
  return node_->gradient ? Tensor(node_->gradient).double_data() : std::vector<double>{};
}

void Tensor::zero_grad() {
  require(defined(), "Undefined tensor");
  node_->gradient.reset();
}

Tensor Tensor::detach() const {
  require(defined(), "Undefined tensor");
  std::shared_ptr<TensorNode> detached = std::make_shared<TensorNode>();
  detached->shape = shape();
  detached->value = node_->value;
  return Tensor(detached);
}

namespace autograd {

void accumulate(const Tensor& tensor, const Tensor& gradient) {
  if (!tensor.requires_grad()) {
    return;
  }
  require(tensor.shape() == gradient.shape(), "Gradient shape mismatch");
  require(tensor.backend() == gradient.backend() && tensor.device() == gradient.device(), "Gradient device mismatch");
  DeviceScope owner(tensor.backend(), tensor.device());
  const Tensor converted = cast(gradient, tensor.dtype());
  std::shared_ptr<TensorNode>& current = tensor.node()->gradient;
  current = current ? add(Tensor(current), converted).node()
                    : converted.detach().node();
}

}  // namespace autograd

void Tensor::backward() {
  DeviceScope owner(backend(), device());
  require(numel() == 1, "Non-scalar backward requires an explicit gradient");
  require(!is_complex(dtype()), "Complex backward requires an explicit complex gradient");
  backward(ones(shape(), backend(), false, dtype()));
}

void Tensor::backward(const Tensor& seed) {
  DeviceScope owner(backend(), device());
  require(requires_grad(), "Tensor does not require gradients");
  require(seed.shape() == shape() && seed.backend() == backend() && seed.device() == device(),
          "Backward seed mismatch");
  require(is_complex(seed.dtype()) == is_complex(dtype()), "Backward seed must match the real or complex output category");
  graph::RecordingGuard no_grad;
  std::vector<std::shared_ptr<TensorNode>> topology = graph::topological_order(*this);
  for (std::shared_ptr<TensorNode>& node : topology) {
    if (!node->parents.empty()) {
      node->gradient.reset();
    }
  }
  autograd::accumulate(*this, cast(seed, dtype()));
  for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
    if ((*it)->gradient && (*it)->backward) {
      DeviceScope node_device((*it)->value->backend, (*it)->value->device_ordinal());
      (*it)->backward(Tensor((*it)->gradient));
    }
  }
}

}  // namespace elman
