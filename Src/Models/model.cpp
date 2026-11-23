#include "Models/Model.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace elman {
std::unique_ptr<Model> create_resnet(const Json& definition, Backend backend, uint32_t seed);
std::unique_ptr<Model> create_spatial_policy_value(const Json& definition, Backend backend, uint32_t seed);
std::unique_ptr<Model> create_continuous_actor_critic(const Json& definition, Backend backend, uint32_t seed);
namespace {

std::vector<int> integers(const Json& object, const std::string& key) {
    std::vector<int> result;
    for (const Json& item : object.at(key).array()) result.push_back(item.integer());
    return result;
}
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("Model: " + message);
}
Tensor active(const Tensor& tensor, bool training) { return training ? tensor : tensor.detach(); }

class AffineModel final : public Model {
public:
    AffineModel(const Json& definition, Backend backend, uint32_t seed, bool regression)
        : Model(definition.at("name").string(), definition.at("architecture").string(), backend, seed), regression_(regression) {
        features_ = definition.at("input_features").integer();
        outputs_ = regression ? definition.int_or("output_features", 1) : definition.at("num_classes").integer();
        require(features_ > 0 && outputs_ >= (regression ? 1 : 2), "invalid affine model dimensions");
        weight_ = parameter("linear.weight", {features_, outputs_}, 0.01f);
        bias_ = constant_parameter("linear.bias", {outputs_}, 0);
    }
    Shape sample_shape() const override { return {features_}; }
    int output_classes() const override { return outputs_; }
    bool regression() const override { return regression_; }
    Tensor forward(const Tensor& input, bool training, uint32_t) override {
        require(input.shape().size() == 2 && input.shape()[1] == features_ && input.dtype() == DType::Float32,
                "Affine model input must be float32 [batch, input_features]");
        return add(matmul(input, active(weight_, training)), active(bias_, training));
    }
private:
    int features_, outputs_;
    bool regression_;
    Tensor weight_, bias_;
};

class AlexNet final : public Model {
public:
    AlexNet(const Json& definition, Backend backend, uint32_t seed)
        : Model(definition.at("name").string(), "alexnet", backend, seed) {
        input_ = integers(definition, "input_shape");
        channels_ = integers(definition, "channels");
        kernels_ = integers(definition, "kernels");
        strides_ = integers(definition, "strides");
        paddings_ = integers(definition, "paddings");
        std::vector<int> hidden = integers(definition, "hidden_features");
        classes_ = definition.at("num_classes").integer();
        pool_kernel_ = definition.int_or("pool_kernel", 3);
        pool_stride_ = definition.int_or("pool_stride", 2);
        dropout_ = static_cast<float>(definition.number_or("dropout", 0.5));
        require(input_.size() == 3 && channels_.size() == 5 && kernels_.size() == 5 && strides_.size() == 5 && paddings_.size() == 5 && hidden.size() == 2,
                "AlexNet requires CHW input, five convolutions and two hidden classifiers");
        require(classes_ > 1 && pool_kernel_ > 0 && pool_stride_ > 0 && dropout_ >= 0 && dropout_ < 1, "invalid AlexNet settings");
        int in_channels = input_[0], height = input_[1], width = input_[2];
        require(in_channels > 0 && height > 0 && width > 0, "input dimensions must be positive");
        for (int i = 0; i < 5; ++i) {
            require(channels_[i] > 0 && kernels_[i] > 0 && strides_[i] > 0 && paddings_[i] >= 0, "invalid convolution dimensions");
            std::string prefix = "features.conv" + std::to_string(i + 1);
            float fan_in = static_cast<float>(in_channels * kernels_[i] * kernels_[i]);
            convolutions_.push_back({parameter(prefix + ".weight", {channels_[i], in_channels, kernels_[i], kernels_[i]}, std::sqrt(2.0f / fan_in)),
                                     constant_parameter(prefix + ".bias", {channels_[i]}, 0.01f)});
            require(height + 2 * paddings_[i] >= kernels_[i] && width + 2 * paddings_[i] >= kernels_[i], "kernel exceeds padded spatial input");
            height = (height + 2 * paddings_[i] - kernels_[i]) / strides_[i] + 1;
            width = (width + 2 * paddings_[i] - kernels_[i]) / strides_[i] + 1;
            require(height > 0 && width > 0, "convolution produces empty spatial output");
            if (i == 0 || i == 1 || i == 4) {
                require(height >= pool_kernel_ && width >= pool_kernel_, "pooling kernel exceeds spatial input");
                height = (height - pool_kernel_) / pool_stride_ + 1;
                width = (width - pool_kernel_) / pool_stride_ + 1;
                require(height > 0 && width > 0, "pooling produces empty spatial output");
            }
            in_channels = channels_[i];
        }
        flat_features_ = channels_.back() * height * width;
        std::vector<int> features = {flat_features_, hidden[0], hidden[1], classes_};
        for (int i = 0; i < 3; ++i) {
            require(features[i] > 0 && features[i + 1] > 0, "classifier dimensions must be positive");
            std::string prefix = "classifier.fc" + std::to_string(i + 1);
            classifiers_.push_back({parameter(prefix + ".weight", {features[i], features[i + 1]}, std::sqrt(2.0f / features[i])),
                                    constant_parameter(prefix + ".bias", {features[i + 1]}, 0.01f)});
        }
    }
    Shape sample_shape() const override { return input_; }
    int output_classes() const override { return classes_; }
    Tensor forward(const Tensor& input, bool training, uint32_t seed) override {
        require(input.shape().size() == 4 && Shape(input.shape().begin() + 1, input.shape().end()) == input_, "AlexNet input must match NCHW model shape");
        Tensor value = input;
        for (int i = 0; i < 5; ++i) {
            value = relu(conv2d(value, active(convolutions_[i].weight, training), active(convolutions_[i].bias, training), strides_[i], paddings_[i]));
            if (i == 0 || i == 1 || i == 4) value = max_pool2d(value, pool_kernel_, pool_stride_);
        }
        value = reshape(value, {input.shape()[0], flat_features_});
        for (int i = 0; i < 3; ++i) {
            if (i < 2) value = dropout(value, dropout_, seed + i, training);
            value = add(matmul(value, active(classifiers_[i].weight, training)), active(classifiers_[i].bias, training));
            if (i < 2) value = relu(value);
        }
        return value;
    }
private:
    struct Affine { Tensor weight, bias; };
    Shape input_;
    std::vector<int> channels_, kernels_, strides_, paddings_;
    std::vector<Affine> convolutions_, classifiers_;
    int classes_, pool_kernel_, pool_stride_, flat_features_;
    float dropout_;
};

class GPT2 final : public Model {
public:
    GPT2(const Json& definition, Backend backend, uint32_t seed)
        : Model(definition.at("name").string(), "gpt2", backend, seed) {
        vocabulary_ = definition.at("vocab_size").integer();
        context_ = definition.at("context_length").integer();
        width_ = definition.at("embedding_dim").integer();
        heads_ = definition.at("num_heads").integer();
        int layer_count = definition.at("num_layers").integer();
        int expansion = definition.int_or("mlp_ratio", 4);
        dropout_ = static_cast<float>(definition.number_or("dropout", 0.1));
        epsilon_ = static_cast<float>(definition.number_or("layer_norm_epsilon", 1e-5));
        require(vocabulary_ > 1 && context_ > 0 && width_ > 0 && heads_ > 0 && width_ % heads_ == 0 && layer_count > 0 && expansion > 0,
                "invalid GPT2 dimensions or head count");
        require(dropout_ >= 0 && dropout_ < 1 && epsilon_ > 0, "invalid GPT2 dropout or epsilon");
        token_embedding_ = parameter("transformer.wte.weight", {vocabulary_, width_}, 0.02f);
        position_embedding_ = parameter("transformer.wpe.weight", {context_, width_}, 0.02f);
        for (int i = 0; i < layer_count; ++i) {
            std::string prefix = "transformer.h." + std::to_string(i);
            Block block;
            block.ln1_weight = constant_parameter(prefix + ".ln_1.weight", {width_}, 1);
            block.ln1_bias = constant_parameter(prefix + ".ln_1.bias", {width_}, 0);
            block.qkv_weight = parameter(prefix + ".attn.c_attn.weight", {width_, 3 * width_}, 0.02f);
            block.qkv_bias = constant_parameter(prefix + ".attn.c_attn.bias", {3 * width_}, 0);
            block.projection_weight = parameter(prefix + ".attn.c_proj.weight", {width_, width_}, 0.02f / std::sqrt(2.0f * layer_count));
            block.projection_bias = constant_parameter(prefix + ".attn.c_proj.bias", {width_}, 0);
            block.ln2_weight = constant_parameter(prefix + ".ln_2.weight", {width_}, 1);
            block.ln2_bias = constant_parameter(prefix + ".ln_2.bias", {width_}, 0);
            block.fc_weight = parameter(prefix + ".mlp.c_fc.weight", {width_, expansion * width_}, 0.02f);
            block.fc_bias = constant_parameter(prefix + ".mlp.c_fc.bias", {expansion * width_}, 0);
            block.mlp_projection_weight = parameter(prefix + ".mlp.c_proj.weight", {expansion * width_, width_}, 0.02f / std::sqrt(2.0f * layer_count));
            block.mlp_projection_bias = constant_parameter(prefix + ".mlp.c_proj.bias", {width_}, 0);
            blocks_.push_back(std::move(block));
        }
        final_weight_ = constant_parameter("transformer.ln_f.weight", {width_}, 1);
        final_bias_ = constant_parameter("transformer.ln_f.bias", {width_}, 0);
    }
    Shape sample_shape() const override { return {context_}; }
    int output_classes() const override { return vocabulary_; }
    Tensor forward(const Tensor& input, bool training, uint32_t seed) override {
        require(input.shape().size() == 2 && input.shape()[1] <= context_ && input.shape()[1] > 0, "GPT2 input must be [batch, sequence] within context length");
        int batch = input.shape()[0], length = input.shape()[1], head_width = width_ / heads_;
        Tensor tokens = reshape(input, {batch * length});
        Tensor value = reshape(index_select(active(token_embedding_, training), 0, tokens), {batch, length, width_});
        std::vector<float> positions(length);
        for (int i = 0; i < length; ++i) positions[i] = static_cast<float>(i);
        Tensor position_ids = Tensor::from_data({length}, positions, backend_, false, DType::Int32);
        value = add(value, index_select(active(position_embedding_, training), 0, position_ids));
        value = dropout(value, dropout_, seed++, training);
        std::vector<float> mask(static_cast<size_t>(length) * length, 0);
        for (int row = 0; row < length; ++row) for (int col = row + 1; col < length; ++col) mask[row * length + col] = -1e9f;
        Tensor causal_mask = Tensor::from_data({1, 1, length, length}, mask, backend_);
        for (GPT2::Block& block : blocks_) {
            Tensor norm = layer_norm(value, active(block.ln1_weight, training), active(block.ln1_bias, training), epsilon_);
            Tensor qkv = add(matmul(norm, active(block.qkv_weight, training)), active(block.qkv_bias, training));
            Tensor query = permute(reshape(slice(qkv, 2, 0, width_), {batch, length, heads_, head_width}), {0, 2, 1, 3});
            Tensor key = permute(reshape(slice(qkv, 2, width_, 2 * width_), {batch, length, heads_, head_width}), {0, 2, 1, 3});
            Tensor val = permute(reshape(slice(qkv, 2, 2 * width_, 3 * width_), {batch, length, heads_, head_width}), {0, 2, 1, 3});
            // Preserve GPT2's reciprocal-multiply scaling, including on CPU.
            Tensor scores = mul(matmul(query, transpose(key, 2, 3)), 1.0f / std::sqrt(static_cast<float>(head_width)));
            Tensor attention = dropout(attention_softmax(scores, causal_mask, 1, 3), dropout_, seed++, training);
            Tensor attended = reshape(permute(matmul(attention, val), {0, 2, 1, 3}), {batch, length, width_});
            Tensor projected = add(matmul(attended, active(block.projection_weight, training)), active(block.projection_bias, training));
            value = add(value, dropout(projected, dropout_, seed++, training));
            norm = layer_norm(value, active(block.ln2_weight, training), active(block.ln2_bias, training), epsilon_);
            Tensor hidden = gelu(add(matmul(norm, active(block.fc_weight, training)), active(block.fc_bias, training)));
            Tensor feedforward = add(matmul(hidden, active(block.mlp_projection_weight, training)), active(block.mlp_projection_bias, training));
            value = add(value, dropout(feedforward, dropout_, seed++, training));
        }
        value = layer_norm(value, active(final_weight_, training), active(final_bias_, training), epsilon_);
        return linear(value, active(token_embedding_, training));
    }
private:
    struct Block {
        Tensor ln1_weight, ln1_bias, qkv_weight, qkv_bias, projection_weight, projection_bias;
        Tensor ln2_weight, ln2_bias, fc_weight, fc_bias, mlp_projection_weight, mlp_projection_bias;
    };
    int vocabulary_, context_, width_, heads_;
    float dropout_, epsilon_;
    Tensor token_embedding_, position_embedding_, final_weight_, final_bias_;
    std::vector<Block> blocks_;
};

}

Model::Model(std::string name, std::string architecture, Backend backend, uint32_t seed)
    : name_(std::move(name)), architecture_(std::move(architecture)), backend_(backend), seed_(seed) {}
Tensor Model::parameter(const std::string& name, const Shape& shape, float standard_deviation) {
    Tensor value = Tensor::normal(shape, 0, standard_deviation, seed_++, backend_, true);
    parameters_.push_back({name, value});
    return value;
}
Tensor Model::constant_parameter(const std::string& name, const Shape& shape, float value) {
    size_t size = 1;
    for (int dim : shape) size *= static_cast<size_t>(dim);
    Tensor tensor = Tensor::from_data(shape, std::vector<float>(size, value), backend_, true);
    parameters_.push_back({name, tensor});
    return tensor;
}
size_t Model::parameter_count() const {
    size_t count = 0;
    for (const Parameter& parameter : parameters_) count += parameter.value.numel();
    return count;
}
std::unique_ptr<Model> create_model(const Json& definition, Backend backend, uint32_t seed) {
    require(definition.at("format").string() == "elman.model", "unsupported model format");
    require(definition.at("version").integer() == 1, "unsupported model version");
    const std::string architecture = definition.at("architecture").string();
    if (architecture == "linear_regression") return std::make_unique<AffineModel>(definition, backend, seed, true);
    if (architecture == "logistic_regression") return std::make_unique<AffineModel>(definition, backend, seed, false);
    if (architecture == "alexnet") return std::make_unique<AlexNet>(definition, backend, seed);
    if (architecture == "gpt2") return std::make_unique<GPT2>(definition, backend, seed);
    if (architecture == "resnet") return create_resnet(definition, backend, seed);
    if (architecture == "spatial_policy_value") return create_spatial_policy_value(definition, backend, seed);
    if (architecture == "continuous_actor_critic") return create_continuous_actor_critic(definition, backend, seed);
    throw std::runtime_error("Unsupported model architecture: " + architecture);
}

}
