#include "Models/Diffusion/Flux2Text.h"
#include "Core/LocalFunction.h"

#include <cmath>
#include <limits>
#include <stdexcept>

#include "Core/Device.h"
#include "IO/Json.h"

namespace elman::flux2 {
TextEncoder::TextEncoder(const std::filesystem::path& directory, DType dtype) : weights_(directory), dtype_(dtype) {
  Json root = Json::read(directory / "config.json");
  qwen_ = root.string_or("model_type", "") == "qwen3";
  if (!qwen_ && root.string_or("model_type", "") != "mistral3") {
    throw std::invalid_argument("Unsupported FLUX.2 text encoder");
  }
  const Json& c = qwen_ ? root : root.at("text_config");
  prefix_ = qwen_ ? "model." : "language_model.model.";
  width_ = c.at("hidden_size").integer();
  intermediate_ = c.at("intermediate_size").integer();
  heads_ = c.at("num_attention_heads").integer();
  kv_heads_ = c.at("num_key_value_heads").integer();
  dimension_ = c.at("head_dim").integer();
  vocab_ = c.at("vocab_size").integer();
  epsilon_ = c.at("rms_norm_eps").number();
  theta_ = c.at("rope_theta").number();
  if (!is_compute_float(dtype_) || width_ <= 0 || intermediate_ <= 0 || heads_ <= 0 || kv_heads_ <= 0 ||
      heads_ % kv_heads_ || dimension_ <= 0 || dimension_ % 2 || int64_t(dimension_) * heads_ > INT32_MAX ||
      vocab_ <= 0 || !(epsilon_ > 0) || !(theta_ > 0) || c.int_or("num_hidden_layers", 0) <= (qwen_ ? 27 : 30) ||
      c.string_or("hidden_act", "silu") != "silu" || c.bool_or("attention_bias", false) ||
      (c.contains("rope_scaling") && !c.at("rope_scaling").is_null()) ||
      (c.contains("sliding_window") && !c.at("sliding_window").is_null())) {
    throw std::invalid_argument("Unsupported FLUX.2 text encoder configuration");
  }
}

Tensor TextEncoder::encode(const Tensor& ids, const Tensor& mask) const {
  const Shape& s = ids.shape();
  if (s.size() != 2 || s != mask.shape() || s[0] < 1 || s[1] < 1 || s[1] > 512 ||
      (ids.dtype() != DType::Int32 && ids.dtype() != DType::Int64) ||
      (mask.dtype() != DType::Int32 && mask.dtype() != DType::Int64 && mask.dtype() != DType::Bool) ||
      mask.backend() != ids.backend() || mask.device() != ids.device()) {
    throw std::invalid_argument("FLUX.2 text encoder needs [B,L] token IDs and a matching padding mask");
  }
  DeviceScope owner(ids.backend(), ids.device());
  if (ids.dtype() == DType::Int64) {
    for (int64_t token : ids.int64_data()) {
      if (token < 0 || token >= vocab_) {
        throw std::invalid_argument("FLUX.2 token ID is outside the vocabulary");
      }
    }
  }
  LocalFunction read{[&](const std::string& name, const Shape& shape) {
    const std::string key = prefix_ + name;
    const SafetensorInfo& info = weights_.info(key);
    if (info.shape != shape || !is_compute_float(info.dtype)) {
      throw std::invalid_argument("Invalid FLUX.2 text weight " + key);
    }
    return cast(weights_.load(key, ids.backend()), dtype_);
  }};
  const int length = s[1], qwidth = heads_ * dimension_, kvwidth = kv_heads_ * dimension_;
  Tensor flat = reshape(cast(ids, DType::Int32), {-1});
  Tensor hidden = reshape(index_select(read("embed_tokens.weight", {vocab_, width_}), 0, flat), {s[0], length, width_});
  std::vector<float> angles(size_t(length) * dimension_);
  for (int pos = 0; pos < length; ++pos) {
    for (int i = 0; i < dimension_; ++i) {
      angles[size_t(pos) * dimension_ + i] = float(pos * std::pow(theta_, -2.0 * (i % (dimension_ / 2)) / dimension_));
    }
  }
  Tensor phase = Tensor::from_data({1, 1, length, dimension_}, angles, ids.backend());
  Tensor cs = cast(cos(phase), dtype_), sn = cast(sin(phase), dtype_);
  Tensor valid = cast(mask, DType::Float32);
  Tensor padding = reshape(mul(sub(valid, 1.0), std::numeric_limits<float>::max()), {s[0], 1, 1, length});
  Tensor visible = reshape(gt(cumsum(valid, 1), Tensor::zeros({}, ids.backend())), {s[0], 1, length, 1});
  Tensor zero = Tensor::zeros({}, ids.backend());
  const int stride = qwen_ ? 9 : 10;
  std::vector<Tensor> states;
  for (int layer = 0; layer < 3 * stride; ++layer) {
    const std::string prefix = "layers." + std::to_string(layer) + ".";
    LocalFunction norm{[&](const Tensor& x, const std::string& name) {
      return rms_norm(x, read(prefix + name + ".weight", {x.shape().back()}), epsilon_, RmsNormRounding::Normalized);
    }};
    LocalFunction project{[&](const Tensor& x, const std::string& name, int out) {
      return linear(x, read(prefix + name + ".weight", {out, x.shape().back()}));
    }};
    Tensor h = norm(hidden, "input_layernorm");
    LocalFunction heads{[&](const std::string& name, int width) {
      Tensor x = project(h, "self_attn." + name + "_proj", width);
      return transpose(reshape(x, {s[0], length, width / dimension_, dimension_}), 1, 2);
    }};
    Tensor q = heads("q", qwidth), k = heads("k", kvwidth), v = heads("v", kvwidth);
    if (qwen_) {
      q = norm(q, "self_attn.q_norm");
      k = norm(k, "self_attn.k_norm");
    }
    q = rotary_embedding(q, cs, sn);
    k = rotary_embedding(k, cs, sn);
    Tensor a = tiled_attention(cast(q, DType::Float32), cast(k, DType::Float32), cast(v, DType::Float32), padding,
                             std::sqrt(float(dimension_)), kAttentionQueryTile, true);
    a = cast(where(visible, a, zero), dtype_);
    a = reshape(transpose(a, 1, 2), {s[0], length, qwidth});
    hidden = add(hidden, project(a, "self_attn.o_proj", width_));
    h = norm(hidden, "post_attention_layernorm");
    Tensor mlp = silu_mul(project(h, "mlp.gate_proj", intermediate_), project(h, "mlp.up_proj", intermediate_));
    hidden = add(hidden, project(mlp, "mlp.down_proj", width_));
    if ((layer + 1) % stride == 0) {
      states.push_back(hidden);
    }
  }
  return concat(states, -1);
}
}  // namespace elman::flux2
