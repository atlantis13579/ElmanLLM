#pragma once

#include "IO/Json.h"
#include "Inference/GraphSession.h"

namespace elman {

// External port names and intermediate graph values have separate identities.
class GraphBuilder {
 public:
  GraphBuilder(Backend backend, int device) : backend_(backend), device_(device) {}
  std::string input(const std::string& name, const TensorSpec& spec);
  std::string constant(Tensor tensor);
  std::string operation(std::string op, const std::vector<std::string>& inputs, Json::Object attributes = {});
  void output(const std::string& name, const std::string& value, const TensorSpec& spec);
  InferenceSession finish() &&;
  Backend backend() const { return backend_; }
  const Tensor* constant_value(const std::string& name) const;

 private:
  std::string next_value();
  Backend backend_;
  int device_;
  size_t counter_ = 0;
  Json::Object inputs_, outputs_;
  Json::Array nodes_;
  NamedTensors weights_;
};

}  // namespace elman
