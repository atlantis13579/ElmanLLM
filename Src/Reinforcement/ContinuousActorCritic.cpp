#include "Models/Model.h"
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
class ContinuousActorCritic final : public Model {
public:
    ContinuousActorCritic(const Json& definition, Backend backend, uint32_t seed)
        : Model(definition.at("name").string(), "continuous_actor_critic", backend, seed),
          features_(definition.at("input_features").integer()),
          actions_(definition.at("actions").integer()) {
        const int width = definition.at("hidden_features").integer();
        const int hidden_layers = definition.int_or("hidden_layers", 3);
        if (features_ <= 0 || actions_ <= 0 || actions_ == std::numeric_limits<int>::max() ||
            width <= 0 || hidden_layers <= 0 || hidden_layers > 16)
            throw std::invalid_argument("Invalid continuous actor-critic dimensions");
        std::vector<int> sizes{features_};
        sizes.insert(sizes.end(), hidden_layers, width);
        sizes.push_back(actions_ + 1);
        for (size_t i = 0; i + 1 < sizes.size(); ++i) {
            weight_.push_back(parameter("layers." + std::to_string(i) + ".weight",
                                        {sizes[i], sizes[i + 1]}, std::sqrt(2.0f / sizes[i])));
            bias_.push_back(constant_parameter("layers." + std::to_string(i) + ".bias", {sizes[i + 1]}, 0));
        }
    }
    Shape sample_shape() const override { return {features_}; }
    int output_classes() const override { return actions_ + 1; }
    bool regression() const override { return true; }
    Tensor forward(const Tensor& input, bool training, uint32_t) override {
        if (input.shape().size() != 2 || input.shape()[1] != features_ || input.dtype() != DType::Float32)
            throw std::invalid_argument("Continuous actor-critic input must be Float32 [batch, features]");
        Tensor x = input;
        for (size_t i = 0; i < weight_.size(); ++i) {
            const Tensor weight = training ? weight_[i] : weight_[i].detach();
            const Tensor bias = training ? bias_[i] : bias_[i].detach();
            x = add(matmul(x, weight), bias);
            if (i + 1 < weight_.size()) x = where(gt(x, Tensor::zeros(x.shape(), x.backend())), x, sub(exp(x), 1));
        }
        return x;
    }
private:
    int features_, actions_;
    std::vector<Tensor> weight_, bias_;
};
} // namespace

std::unique_ptr<Model> create_continuous_actor_critic(const Json& definition, Backend backend, uint32_t seed) {
    return std::make_unique<ContinuousActorCritic>(definition, backend, seed);
}
} // namespace elman
