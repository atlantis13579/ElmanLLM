#pragma once

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "Core/Tensor.h"

namespace elman::nn {

struct NamedTensor {
  std::string name;
  Tensor value;
};

using StateDict = std::map<std::string, Tensor>;

class Module {
 public:
  virtual ~Module() = default;
  Module(const Module&) = delete;
  Module& operator=(const Module&) = delete;

  Tensor operator()(const Tensor& input, uint32_t seed = 1) { return forward(input, seed); }
  bool training() const { return training_; }
  void train(bool enabled = true);
  void eval() { train(false); }
  std::vector<NamedTensor> named_parameters() const;
  std::vector<NamedTensor> named_buffers(bool persistent_only = false) const;
  // The state dictionary owns independent CPU tensors and includes persistent buffers.
  StateDict state_dict() const;
  void load_state_dict(const StateDict& state);

 protected:
  Module() = default;
  virtual Tensor forward(const Tensor& input, uint32_t seed) = 0;
  Tensor register_parameter(std::string name, Tensor value);
  Tensor register_buffer(std::string name, Tensor value, bool persistent = true);
  Module& register_module(std::string name, std::unique_ptr<Module> child);

 private:
  struct Buffer {
    NamedTensor tensor;
    bool persistent;
  };
  struct Child {
    std::string name;
    std::unique_ptr<Module> module;
  };
  void validate_name(const std::string& name) const;
  void collect(std::vector<NamedTensor>& result, const std::string& prefix,
               bool parameters, bool persistent_only) const;
  std::vector<NamedTensor> parameters_;
  std::vector<Buffer> buffers_;
  std::vector<Child> children_;
  bool training_ = true;
};

}  // namespace elman::nn
