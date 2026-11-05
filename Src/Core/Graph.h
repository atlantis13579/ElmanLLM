#pragma once

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "Core/Tensor.h"

namespace elman {

struct TensorNode {
  Shape shape;
  BufferPtr value;
  bool requires_grad = false;
  std::shared_ptr<TensorNode> gradient;
  std::vector<std::shared_ptr<TensorNode>> parents;
  std::vector<uint64_t> versions;
  std::function<void(const Tensor&)> backward;
  std::string operation = "leaf";
  uint64_t saved_version = 0;
};

namespace graph {

class RecordingGuard {
 public:
  RecordingGuard();
  ~RecordingGuard();
  RecordingGuard(const RecordingGuard&) = delete;
  RecordingGuard& operator=(const RecordingGuard&) = delete;

 private:
  bool previous_;
};

Tensor tracked(Tensor output, const std::vector<Tensor>& parents,
               std::function<void(const Tensor&)> backward);
std::vector<std::shared_ptr<TensorNode>> topological_order(
    const Tensor& output);

}  // namespace graph
}  // namespace elman
