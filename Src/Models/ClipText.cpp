#include "Models/ClipText.h"

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "Core/Device.h"
#include "Models/ComponentLayers.h"
#include "IO/Json.h"
#include "Operators/inference_ops.h"

namespace elman {
namespace {

using inference::Attention;
using inference::LayerNorm;
using inference::Linear;
using io::ComponentWeights;

struct ClipConfig {
  int vocabulary, length, width, intermediate, heads, layers;
  double epsilon;
  bool quick_gelu;
  explicit ClipConfig(const Json& config)
      : vocabulary(config.at("vocab_size").integer()), length(config.at("max_position_embeddings").integer()),
        width(config.at("hidden_size").integer()), intermediate(config.at("intermediate_size").integer()),
        heads(config.at("num_attention_heads").integer()), layers(config.at("num_hidden_layers").integer()),
        epsilon(config.number_or("layer_norm_eps", 1e-5)), quick_gelu(config.string_or("hidden_act", "quick_gelu") == "quick_gelu") {
    if (vocabulary < 2 || length < 2 || width < 1 || intermediate < 1 || heads < 1 || width % heads ||
        layers < 1 || !(epsilon > 0) || !std::isfinite(epsilon) ||
        (!quick_gelu && config.at("hidden_act").string() != "gelu"))
      throw std::invalid_argument("Unsupported CLIP text configuration");
  }
};

struct ClipLayer {
  LayerNorm attention_norm, mlp_norm;
  Attention attention;
  Linear first, second;
  ClipLayer(const ComponentWeights& weights, const ClipConfig& config, const std::string& prefix)
      : attention_norm(model_weights::layer_norm(weights, prefix + ".layer_norm1", config.width, config.epsilon)),
        mlp_norm(model_weights::layer_norm(weights, prefix + ".layer_norm2", config.width, config.epsilon)),
        attention(model_weights::linear(weights, prefix + ".self_attn.q_proj", config.width, config.width),
                  model_weights::linear(weights, prefix + ".self_attn.k_proj", config.width, config.width),
                  model_weights::linear(weights, prefix + ".self_attn.v_proj", config.width, config.width),
                  model_weights::linear(weights, prefix + ".self_attn.out_proj", config.width, config.width), config.heads),
        first(model_weights::linear(weights, prefix + ".mlp.fc1", config.width, config.intermediate)),
        second(model_weights::linear(weights, prefix + ".mlp.fc2", config.intermediate, config.width)) {}
  Tensor forward(const Tensor& input, bool use_quick_gelu) const {
    auto normalized = attention_norm(input);
    auto hidden = add(input, attention(normalized, normalized, true));
    auto value = first(mlp_norm(hidden));
    value = use_quick_gelu ? quick_gelu(value) : exact_gelu(value);
    return add(hidden, second(value));
  }
};

struct ClipText {
  ClipConfig config;
  Tensor embedding, positions;
  LayerNorm final_norm;
  std::vector<ClipLayer> layers;
  ClipText(ClipConfig config_, const ComponentWeights& weights)
      : config(config_), embedding(weights.get("text_model.embeddings.token_embedding.weight", {config.vocabulary, config.width})),
        positions(weights.get("text_model.embeddings.position_embedding.weight", {config.length, config.width})),
        final_norm(model_weights::layer_norm(weights, "text_model.final_layer_norm", config.width, config.epsilon)) {
    layers.reserve(config.layers);
    for (int layer = 0; layer < config.layers; ++layer)
      layers.emplace_back(weights, config, "text_model.encoder.layers." + std::to_string(layer));
  }
  Tensor forward(const Tensor& ids) const {
    const int batch = ids.shape()[0];
    if (batch < 1 || batch > std::numeric_limits<int>::max() / config.length)
      throw std::invalid_argument("Invalid CLIP text batch");
    auto tokens = index_select(embedding, 0, reshape(ids, {batch * config.length}));
    auto hidden = add(reshape(tokens, {batch, config.length, config.width}), positions);
    for (const auto& layer : layers) hidden = layer.forward(hidden, config.quick_gelu);
    return final_norm(hidden);
  }
};

}  // namespace

InferenceSession load_clip_text_encoder(const std::filesystem::path& directory, Backend backend,
                                         int device, DType dtype) {
  DeviceScope owner(backend, device);
  const ClipConfig config(Json::read((directory / "config.json").string()));
  auto model = std::make_shared<ClipText>(config, ComponentWeights(directory, "model", backend, dtype));
  return InferenceSession(backend, device, {{"input_ids", {DType::Int32, {-1, config.length}}}},
      {{"last_hidden_state", {dtype, {-1, config.length, config.width}}}},
      [model = std::move(model)](const NamedTensors& inputs) {
        return NamedTensors{{"last_hidden_state", model->forward(inputs.at("input_ids"))}};
      });
}

}  // namespace elman
