#pragma once
#include "Layers/GraphModule.h"
#include "Models/Model.h"

namespace elman {
class GraphModel final : public Model {
 public:
  GraphModel(const Json& definition, Backend backend, uint32_t seed);
  Tensor forward(const Tensor& input, bool training,
                 uint32_t seed = 1) override;
  nn::TensorMap forward(const nn::TensorMap& inputs, bool training,
                        uint32_t seed = 1);
  Shape sample_shape() const override {
    return sample_shape_;
  }
  int output_classes() const override {
    return classes_;
  }
  bool regression() const override {
    return task_ == "regression";
  }
  bool language() const override {
    return task_ == "language_modeling";
  }
  std::vector<Parameter> checkpoint_buffers() const override;
  const Json& definition() const {
    return definition_;
  }

 private:
  Json definition_;
  std::unique_ptr<nn::GraphModule> graph_;
  Shape sample_shape_;
  std::string task_;
  int classes_;
};
}  // namespace elman
