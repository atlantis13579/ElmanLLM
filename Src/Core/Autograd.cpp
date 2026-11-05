#include "Core/Autograd.h"

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

void Tensor::zero_grad() {
  require(defined(), "Undefined tensor");
  node_->gradient.reset();
}

Tensor Tensor::detach() const {
  require(defined(), "Undefined tensor");
  auto detached = std::make_shared<TensorNode>();
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
  auto& current = tensor.node()->gradient;
  current = current ? add(Tensor(current), gradient).node()
                    : gradient.detach().node();
}

}  // namespace autograd

void Tensor::backward() {
  require(numel() == 1, "Non-scalar backward requires an explicit gradient");
  backward(ones(shape(), backend()));
}

void Tensor::backward(const Tensor& seed) {
  require(requires_grad(), "Tensor does not require gradients");
  require(seed.shape() == shape() && seed.backend() == backend(),
          "Backward seed mismatch");
  graph::RecordingGuard no_grad;
  auto topology = graph::topological_order(*this);
  for (auto& node : topology) {
    if (!node->parents.empty()) {
      node->gradient.reset();
    }
  }
  autograd::accumulate(*this, seed);
  for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
    if ((*it)->gradient && (*it)->backward) {
      (*it)->backward(Tensor((*it)->gradient));
    }
  }
}

}  // namespace elman
