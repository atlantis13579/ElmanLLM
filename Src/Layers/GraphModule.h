#pragma once

#include "IO/Json.h"
#include "Layers/Module.h"

namespace elman::nn {

using TensorMap = std::map<std::string, Tensor>;

// A reusable layer DAG. Sequence is the single-input/single-output shorthand.
class GraphModule final : public Module {
 public:
  GraphModule(const Json& definition, Backend backend, uint32_t& seed);
  TensorMap run(const TensorMap& inputs, uint32_t seed = 1);

 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  struct Node {
    std::string name, type;
    Json config;
    std::vector<std::string> inputs, outputs;
    Module* module = nullptr;
  };
  std::vector<std::string> inputs_, outputs_;
  std::vector<Node> nodes_;
  std::map<std::string, size_t> readers_;
  Backend backend_;
};

std::unique_ptr<Module> create_layer(const std::string& type,
                                     const Json& config, Backend backend,
                                     uint32_t& seed);
const Json& operator_catalog();
void validate_operation(const std::string& type, const Json& config,
                        size_t inputs, size_t outputs);
std::vector<Tensor> execute_operation(const std::string& type,
                                      const std::vector<Tensor>& inputs,
                                      const Json& config, Backend backend,
                                      uint32_t seed, bool training);
Json resolve_model_config(const Json& definition);

}  // namespace elman::nn
