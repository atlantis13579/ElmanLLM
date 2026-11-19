#include "Models/ModuleModel.h"

#include <stdexcept>
#include <utility>

#include "Core/Graph.h"
#include "Core/TensorDispatch.h"

namespace elman {

ModuleModel::ModuleModel(std::string name, std::string architecture, Backend backend,
                         Shape input_shape, int outputs, std::unique_ptr<nn::Module> module)
    : Model(std::move(name), std::move(architecture), backend, 1), input_shape_(std::move(input_shape)),
      outputs_(outputs), module_(std::move(module)) {
  if (!module_ || outputs_ <= 0) throw std::invalid_argument("Invalid module model");
  shape_element_count(input_shape_);
  for (auto& parameter : module_->named_parameters()) {
    if (parameter.value.backend() != backend) throw std::invalid_argument("Module parameter backend mismatch");
    if (parameter.value.requires_grad()) parameters_.push_back({std::move(parameter.name), std::move(parameter.value)});
  }
}

Tensor ModuleModel::forward(const Tensor& input, bool training, uint32_t seed) {
  module_->train(training);
  if (training) return (*module_)(input, seed);
  graph::RecordingGuard no_grad;
  return (*module_)(input, seed).detach();
}

std::vector<Parameter> ModuleModel::checkpoint_buffers() const {
  std::vector<Parameter> state;
  for (auto& parameter : module_->named_parameters()) {
    if (!parameter.value.requires_grad()) state.push_back({std::move(parameter.name), std::move(parameter.value)});
  }
  for (auto& buffer : module_->named_buffers(true)) {
    state.push_back({std::move(buffer.name), std::move(buffer.value)});
  }
  return state;
}

}  // namespace elman
