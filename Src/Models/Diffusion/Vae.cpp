#include "Models/Diffusion/Vae.h"

#include <cmath>
#include <memory>
#include <stdexcept>

#include "Core/Device.h"
#include "Models/Diffusion/ModelConfig.h"
#include "Models/Diffusion/Blocks.h"
#include "Operators/inference_ops.h"

namespace elman {
namespace {
using diffusion::block_channels;
using diffusion::ResnetBlock;
using io::ComponentWeights;
using namespace inference;

struct VaeConfig {
  std::vector<int> channels;
  int latent, output, groups, repeats;
  double scaling;
  bool quantize, attention;
  explicit VaeConfig(const Json& config)
      : channels(block_channels(config)), latent(config.int_or("latent_channels", 4)),
        output(config.int_or("out_channels", 3)), groups(config.int_or("norm_num_groups", 32)),
        repeats(config.int_or("layers_per_block", 2)), scaling(config.number_or("scaling_factor", 0.18215)),
        quantize(config.bool_or("use_post_quant_conv", true)), attention(config.bool_or("mid_block_add_attention", true)) {
    if (latent <= 0 || output <= 0 || groups <= 0 || repeats < 1 || repeats > 16 ||
        !(scaling > 0) || !std::isfinite(scaling) || config.string_or("act_fn", "silu") != "silu")
      throw std::invalid_argument("Unsupported Stable Diffusion VAE configuration");
    const auto& types = config.at("up_block_types").array();
    if (types.size() != channels.size()) throw std::invalid_argument("VAE up blocks do not match channel stages");
    for (const auto& type : types) if (type.string() != "UpDecoderBlock2D")
      throw std::invalid_argument("Unsupported VAE upsampling block");
  }
};

struct VaeAttention {
  GroupNorm norm;
  Attention attention;
  VaeAttention(const ComponentWeights& weights, const std::string& prefix, int channels, int groups)
      : norm(model_weights::group_norm(weights, prefix + ".group_norm", channels, groups, 1e-6)),
        attention(load(weights, prefix, channels)) {}
  static Attention load(const ComponentWeights& weights, const std::string& prefix, int channels) {
    const bool modern = weights.contains(prefix + ".to_q.weight");
    auto projection = [&](const char* current, const char* legacy) {
      return model_weights::linear(weights, prefix + (modern ? current : legacy), channels, channels, true, true);
    };
    return Attention(projection(".to_q", ".query"), projection(".to_k", ".key"),
                     projection(".to_v", ".value"), projection(".to_out.0", ".proj_attn"), 1);
  }
  Tensor operator()(const Tensor& input) const {
    auto x = image_tokens(norm(input));
    return add(input, token_image(attention(x, x), input.shape()[2], input.shape()[3]));
  }
};

struct VaeStage {
  std::vector<ResnetBlock> blocks;
  std::optional<Conv2d> resize;
};

struct VaeModel {
  VaeConfig config;
  Conv2d input, output;
  GroupNorm final_norm;
  ResnetBlock mid_first, mid_second;
  std::optional<Conv2d> post_quant;
  std::optional<VaeAttention> mid_attention;
  std::vector<VaeStage> stages;
  VaeModel(VaeConfig config_, const ComponentWeights& weights)
      : config(std::move(config_)), input(model_weights::conv2d(weights, "decoder.conv_in", config.latent, config.channels.back(), 3, 1, 1)),
        output(model_weights::conv2d(weights, "decoder.conv_out", config.channels.front(), config.output, 3, 1, 1)),
        final_norm(model_weights::group_norm(weights, "decoder.conv_norm_out", config.channels.front(), config.groups, 1e-6)),
        mid_first(weights, "decoder.mid_block.resnets.0", config.channels.back(), config.channels.back(), config.groups, 1e-6),
        mid_second(weights, "decoder.mid_block.resnets.1", config.channels.back(), config.channels.back(), config.groups, 1e-6) {
    if (config.quantize) post_quant = model_weights::conv2d(weights, "post_quant_conv", config.latent, config.latent, 1);
    if (config.attention) mid_attention.emplace(weights, "decoder.mid_block.attentions.0", config.channels.back(), config.groups);
    int current = config.channels.back();
    for (size_t level = 0; level < config.channels.size(); ++level) {
      const auto prefix = "decoder.up_blocks." + std::to_string(level);
      const int width = config.channels[config.channels.size() - level - 1];
      auto& stage = stages.emplace_back();
      for (int block = 0; block <= config.repeats; ++block) {
        stage.blocks.emplace_back(weights, prefix + ".resnets." + std::to_string(block), current, width, config.groups, 1e-6);
        current = width;
      }
      if (level + 1 < config.channels.size()) stage.resize = model_weights::conv2d(weights, prefix + ".upsamplers.0.conv", width, width, 3, 1, 1);
    }
  }
  Tensor forward(const Tensor& latent) const {
    const auto& shape = latent.shape();
    if (shape[0] <= 0 || shape[2] <= 0 || shape[3] <= 0) throw std::invalid_argument("Invalid VAE latent shape");
    auto x = mid_first(input(post_quant ? (*post_quant)(latent) : latent));
    if (mid_attention) x = (*mid_attention)(x);
    x = mid_second(x);
    for (const auto& stage : stages) {
      for (const auto& block : stage.blocks) x = block(x);
      if (stage.resize) x = (*stage.resize)(upsample_nearest2d(x));
    }
    return output(silu(final_norm(x)));
  }
};

}  // namespace

VaeDecoder load_diffusion_vae(const std::filesystem::path& directory, Backend backend, int device, DType dtype) {
  DeviceScope owner(backend, device);
  const VaeConfig config(Json::read((directory / "config.json").string()));
  auto model = std::make_shared<VaeModel>(config, ComponentWeights(directory, "diffusion_pytorch_model", backend, dtype));
  return {InferenceSession(backend, device, {{"latent", {dtype, {-1, config.latent, -1, -1}}}},
      {{"sample", {dtype, {-1, config.output, -1, -1}}}}, [model = std::move(model)](const NamedTensors& inputs) {
        return NamedTensors{{"sample", model->forward(inputs.at("latent"))}};
      }), config.scaling, 1 << (config.channels.size() - 1)};
}

}  // namespace elman
