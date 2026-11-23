#include "Models/ModuleModel.h"

#include <algorithm>
#include <stdexcept>

#include "Layers/Layers.h"

namespace elman {
namespace {

class ResidualBlock final : public nn::Module {
 public:
  ResidualBlock(int inputs, int outputs, int stride, int groups, Backend backend, uint32_t& seed) {
    std::unique_ptr<nn::Sequential> path = std::make_unique<nn::Sequential>();
    path->append("conv1", std::make_unique<nn::Conv2d>(inputs, outputs, 3, backend, seed++, stride, 1, false));
    path->append("norm1", std::make_unique<nn::GroupNorm>(groups, outputs, backend));
    path->append("relu", std::make_unique<nn::ReLU>());
    path->append("conv2", std::make_unique<nn::Conv2d>(outputs, outputs, 3, backend, seed++, 1, 1, false));
    path->append("norm2", std::make_unique<nn::GroupNorm>(groups, outputs, backend));
    path_ = &register_module("path", std::move(path));
    if (inputs != outputs || stride != 1) {
      std::unique_ptr<nn::Sequential> skip = std::make_unique<nn::Sequential>();
      skip->append("conv", std::make_unique<nn::Conv2d>(inputs, outputs, 1, backend, seed++, stride, 0, false));
      skip->append("norm", std::make_unique<nn::GroupNorm>(groups, outputs, backend));
      skip_ = &register_module("skip", std::move(skip));
    }
  }

 private:
  Tensor forward(const Tensor& input, uint32_t seed) override {
    const Tensor skip = skip_ ? (*skip_)(input, seed) : input;
    return relu(add((*path_)(input, seed), skip));
  }
  nn::Module* path_;
  nn::Module* skip_ = nullptr;
};

class ResNet final : public nn::Module {
 public:
  ResNet(int input_channels, const std::vector<int>& channels, const std::vector<int>& repeats,
         int groups, int classes, Backend backend, uint32_t seed) {
    std::unique_ptr<nn::Sequential> features = std::make_unique<nn::Sequential>();
    features->append("stem_conv", std::make_unique<nn::Conv2d>(input_channels, channels.front(), 3, backend, seed++, 1, 1, false));
    features->append("stem_norm", std::make_unique<nn::GroupNorm>(groups, channels.front(), backend));
    features->append("stem_relu", std::make_unique<nn::ReLU>());
    int previous = channels.front();
    for (size_t stage = 0; stage < channels.size(); ++stage) {
      for (int block = 0; block < repeats[stage]; ++block) {
        const int stride = stage > 0 && block == 0 ? 2 : 1;
        features->append("stage" + std::to_string(stage + 1) + "_block" + std::to_string(block + 1),
                         std::make_unique<ResidualBlock>(previous, channels[stage], stride, groups, backend, seed));
        previous = channels[stage];
      }
    }
    features_ = &register_module("features", std::move(features));
    classifier_ = &register_module("classifier", std::make_unique<nn::Linear>(previous, classes, backend, seed));
  }

 private:
  Tensor forward(const Tensor& input, uint32_t seed) override {
    if (input.shape().size() != 4) throw std::invalid_argument("ResNet expects NCHW input");
    Tensor features = (*features_)(input, seed);
    return (*classifier_)(mean(mean(features, 3), 2), seed);
  }
  nn::Module* features_;
  nn::Module* classifier_;
};

std::vector<int> positive_list(const Json& definition, const char* key) {
  std::vector<int> result;
  for (const Json& value : definition.at(key).array()) {
    const int number = value.integer();
    if (number <= 0) throw std::invalid_argument(std::string("ResNet dimensions must be positive: ") + key);
    result.push_back(number);
  }
  if (result.empty()) throw std::invalid_argument(std::string("Missing ResNet dimensions: ") + key);
  return result;
}

}  // namespace

std::unique_ptr<Model> create_resnet(const Json& definition, Backend backend, uint32_t seed) {
  const std::vector<int> input = positive_list(definition, "input_shape");
  const std::vector<int> channels = positive_list(definition, "channels");
  const std::vector<int> repeats = positive_list(definition, "blocks");
  const int groups = definition.int_or("groups", 1);
  const int classes = definition.at("num_classes").integer();
  if (input.size() != 3 || channels.size() != repeats.size() || classes < 2 || groups < 1 ||
      std::any_of(channels.begin(), channels.end(), [groups](int width) { return width % groups != 0; }))
    throw std::invalid_argument("Invalid ResNet model configuration");
  std::unique_ptr<ResNet> module = std::make_unique<ResNet>(input[0], channels, repeats, groups, classes, backend, seed);
  return std::make_unique<ModuleModel>(definition.at("name").string(), "resnet", backend, input, classes, std::move(module));
}

}  // namespace elman
