#include "Models/ModuleModel.h"
#include "Core/LocalFunction.h"
#include "Layers/Layers.h"
#include <limits>
#include <stdexcept>

namespace elman {
namespace {

class SpatialPolicyValue final : public nn::Module {
public:
    SpatialPolicyValue(int channels, int width, int blocks, int global_actions, Backend backend, uint32_t seed)
        : channels_(channels) {
        LocalFunction conv_norm{[&](int inputs, int kernel) {
            std::unique_ptr<nn::Sequential> path = std::make_unique<nn::Sequential>();
            path->append("conv", std::make_unique<nn::Conv2d>(inputs, width, kernel, backend, seed++, 1, kernel / 2, false));
            path->append("norm", std::make_unique<nn::GroupNorm>(1, width, backend));
            return path;
        }};
        stem_ = &register_module("stem", conv_norm(channels, 3));
        for (int i = 0; i < blocks; ++i) {
            std::unique_ptr<nn::Sequential> block = std::make_unique<nn::Sequential>();
            block->append("first", conv_norm(width, 3));
            block->append("relu", std::make_unique<nn::ReLU>());
            block->append("second", conv_norm(width, 3));
            blocks_.push_back(&register_module("block" + std::to_string(i), std::move(block)));
        }
        local_ = &register_module("local_policy", std::make_unique<nn::Conv2d>(width, 1, 1, backend, seed++));
        global_ = &register_module("global_policy", std::make_unique<nn::Linear>(width, global_actions, backend, seed++));
        value_ = &register_module("value", std::make_unique<nn::Linear>(width, 1, backend, seed++));
    }
private:
    Tensor forward(const Tensor& input, uint32_t seed) override {
        const Shape& shape = input.shape();
        if (shape.size() != 4 || shape[1] != channels_ || input.numel() == 0)
            throw std::invalid_argument("Spatial policy-value expects nonempty NCHW observations");
        Tensor x = relu((*stem_)(input, seed));
        for (nn::Module* block : blocks_) x = relu(add(x, (*block)(x, seed)));
        Tensor local = reshape((*local_)(x, seed), {shape[0], -1});
        Tensor pooled = mean(mean(x, 3), 2);
        return concat({local, (*global_)(pooled, seed), tanh((*value_)(pooled, seed))}, 1);
    }
    int channels_;
    nn::Module *stem_, *local_, *global_, *value_;
    std::vector<nn::Module*> blocks_;
};

class DenseSpatialPolicyValue final : public nn::Module {
public:
    DenseSpatialPolicyValue(int channels, int height, int width, int filters, int blocks,
                            int policy_channels, int value_hidden, int global_actions,
                            Backend backend, uint32_t seed)
        : channels_(channels), height_(height), width_(width) {
        auto conv_bn = [&](int inputs, int outputs, int kernel) {
            std::unique_ptr<nn::Sequential> layers = std::make_unique<nn::Sequential>();
            layers->append("conv", std::make_unique<nn::Conv2d>(inputs, outputs, kernel, backend, seed++,
                                                                  1, kernel / 2));
            layers->append("bn", std::make_unique<nn::BatchNorm2d>(outputs, backend));
            return layers;
        };
        std::unique_ptr<nn::Sequential> stem = conv_bn(channels, filters, 3);
        stem->append("relu", std::make_unique<nn::ReLU>());
        stem_ = &register_module("dense_stem", std::move(stem));
        for (int i = 0; i < blocks; ++i) {
            std::unique_ptr<nn::Sequential> block = std::make_unique<nn::Sequential>();
            block->append("first", conv_bn(filters, filters, 3));
            block->append("relu", std::make_unique<nn::ReLU>());
            block->append("second", conv_bn(filters, filters, 3));
            blocks_.push_back(&register_module("dense_block" + std::to_string(i), std::move(block)));
        }
        policy_conv_ = &register_module("dense_policy_conv", std::make_unique<nn::Conv2d>(filters, policy_channels,
                                                                                             1, backend, seed++));
        policy_norm_ = &register_module("dense_policy_bn", std::make_unique<nn::BatchNorm2d>(policy_channels, backend));
        policy_linear_ = &register_module("dense_policy_linear", std::make_unique<nn::Linear>(
            policy_channels * height * width, height * width + global_actions, backend, seed++));
        value_conv_ = &register_module("dense_value_conv", std::make_unique<nn::Conv2d>(filters, 1, 1, backend, seed++));
        value_norm_ = &register_module("dense_value_bn", std::make_unique<nn::BatchNorm2d>(1, backend));
        value_linear1_ = &register_module("dense_value_linear1", std::make_unique<nn::Linear>(
            height * width, value_hidden, backend, seed++));
        value_linear2_ = &register_module("dense_value_linear2", std::make_unique<nn::Linear>(
            value_hidden, 1, backend, seed++));
    }
private:
    Tensor forward(const Tensor& input, uint32_t seed) override {
        const Shape& shape = input.shape();
        if (shape.size() != 4 || shape[1] != channels_ || shape[2] != height_ || shape[3] != width_ || !shape[0])
            throw std::invalid_argument("Dense spatial policy-value expects its configured NCHW shape");
        Tensor x = (*stem_)(input, seed);
        for (nn::Module* block : blocks_) x = relu(add(x, (*block)(x, seed)));
        Tensor policy = relu((*policy_norm_)((*policy_conv_)(x, seed), seed));
        policy = (*policy_linear_)(reshape(policy, {shape[0], -1}), seed);
        Tensor value = relu((*value_norm_)((*value_conv_)(x, seed), seed));
        value = relu((*value_linear1_)(reshape(value, {shape[0], -1}), seed));
        value = tanh((*value_linear2_)(value, seed));
        return concat({policy, value}, 1);
    }
    int channels_, height_, width_;
    nn::Module *stem_, *policy_conv_, *policy_norm_, *policy_linear_;
    nn::Module *value_conv_, *value_norm_, *value_linear1_, *value_linear2_;
    std::vector<nn::Module*> blocks_;
};

} // namespace

std::unique_ptr<Model> create_spatial_policy_value(const Json& definition, Backend backend, uint32_t seed) {
    Shape sample;
    for (const Json& dim : definition.at("input_shape").array()) sample.push_back(dim.integer());
    int width = definition.at("channels").integer();
    int blocks = definition.at("blocks").integer();
    int global = definition.at("global_actions").integer();
    if (sample.size() != 3 || sample[0] < 1 || sample[1] < 1 || sample[2] < 1 ||
        width < 1 || blocks < 0 || global < 1 ||
        int64_t(sample[1]) * sample[2] + global + 1 > std::numeric_limits<int>::max())
        throw std::invalid_argument("Invalid spatial policy-value geometry");
    int outputs = sample[1] * sample[2] + global + 1;
    const std::string variant = definition.string_or("variant", "pooled");
    if (variant == "dense") {
        const int policy_channels = definition.int_or("policy_channels", 2);
        const int value_hidden = definition.int_or("value_hidden", 256);
        if (policy_channels < 1 || value_hidden < 1 ||
            int64_t(policy_channels) * sample[1] * sample[2] > std::numeric_limits<int>::max())
            throw std::invalid_argument("Invalid dense spatial policy-value head geometry");
        auto module = std::make_unique<DenseSpatialPolicyValue>(sample[0], sample[1], sample[2], width, blocks,
                                                                 policy_channels, value_hidden, global, backend, seed);
        return std::make_unique<ModuleModel>(definition.at("name").string(), "spatial_policy_value_dense",
                                             backend, sample, outputs, std::move(module));
    }
    if (variant != "pooled") throw std::invalid_argument("Unknown spatial policy-value variant");
    std::unique_ptr<SpatialPolicyValue> module = std::make_unique<SpatialPolicyValue>(sample[0], width, blocks, global, backend, seed);
    return std::make_unique<ModuleModel>(definition.at("name").string(), "spatial_policy_value",
                                         backend, sample, outputs, std::move(module));
}

} // namespace elman
