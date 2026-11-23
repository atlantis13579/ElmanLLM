#pragma once

#include "Models/Model.h"
#include "Layers/Module.h"

namespace elman {

class ModuleModel final : public Model {
 public:
  ModuleModel(std::string name, std::string architecture, Backend backend,
              Shape input_shape, int outputs, std::unique_ptr<nn::Module> module);
  Tensor forward(const Tensor& input, bool training, uint32_t seed = 1) override;
  Shape sample_shape() const override { return input_shape_; }
  int output_classes() const override { return outputs_; }
  std::vector<Parameter> checkpoint_buffers() const override;
  nn::Module& module() { return *module_; }

 private:
  Shape input_shape_;
  int outputs_;
  std::unique_ptr<nn::Module> module_;
};

}  // namespace elman
