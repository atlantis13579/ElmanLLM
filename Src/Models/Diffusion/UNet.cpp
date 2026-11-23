#include "Models/Diffusion/UNet.h"

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
using diffusion::SpatialTransformer;
using io::ComponentWeights;
using namespace inference;

struct UNetConfig {
  std::vector<int> channels;
  std::vector<bool> down_attention, up_attention;
  int in, out, context, groups, repeats, heads, time_width;
  double epsilon, frequency_shift;
  bool flip;
  explicit UNetConfig(const Json& config)
      : channels(block_channels(config)), in(config.at("in_channels").integer()), out(config.at("out_channels").integer()),
        context(config.at("cross_attention_dim").integer()), groups(config.int_or("norm_num_groups", 32)),
        repeats(config.int_or("layers_per_block", 2)), heads(config.int_or("attention_head_dim", 8)),
        time_width(channels.front() * 4), epsilon(config.number_or("norm_eps", 1e-5)),
        frequency_shift(config.number_or("freq_shift", 0)), flip(config.bool_or("flip_sin_to_cos", true)) {
    if (in <= 0 || in != out || context <= 0 || repeats < 1 || repeats > 16 || groups <= 0 || heads <= 0 ||
        channels.front() / 2 <= frequency_shift || !(epsilon > 0) ||
        config.string_or("act_fn", "silu") != "silu" || config.bool_or("center_input_sample", false) ||
        config.bool_or("use_linear_projection", false) || config.bool_or("only_cross_attention", false) ||
        config.int_or("downsample_padding", 1) != 1 || config.int_or("transformer_layers_per_block", 1) != 1 ||
        config.number_or("mid_block_scale_factor", 1) != 1 || config.number_or("resnet_out_scale_factor", 1) != 1 ||
        config.string_or("mid_block_type", "UNetMidBlock2DCrossAttn") != "UNetMidBlock2DCrossAttn")
      throw std::invalid_argument("Unsupported Stable Diffusion U-Net configuration");
    for (const Json& value : config.at("down_block_types").array()) {
      const std::string type = value.string();
      if (type != "CrossAttnDownBlock2D" && type != "DownBlock2D") throw std::invalid_argument("Unknown U-Net down block " + type);
      down_attention.push_back(type == "CrossAttnDownBlock2D");
    }
    for (const Json& value : config.at("up_block_types").array()) {
      const std::string type = value.string();
      if (type != "CrossAttnUpBlock2D" && type != "UpBlock2D") throw std::invalid_argument("Unknown U-Net up block " + type);
      up_attention.push_back(type == "CrossAttnUpBlock2D");
    }
    if (down_attention.size() != channels.size() || up_attention.size() != channels.size())
      throw std::invalid_argument("U-Net block types do not match channel stages");
  }
};

struct UNetBlock {
  ResnetBlock resnet;
  std::optional<SpatialTransformer> attention;
  UNetBlock(const ComponentWeights& weights, const std::string& prefix, int layer, int input, int output,
            bool use_attention, const UNetConfig& config)
      : resnet(weights, prefix + ".resnets." + std::to_string(layer), input, output,
               config.groups, config.epsilon, config.time_width) {
    if (use_attention) attention.emplace(weights, prefix + ".attentions." + std::to_string(layer), output,
                                         config.context, config.heads, config.groups);
  }
  Tensor operator()(const Tensor& input, const Tensor& context, const Tensor& time) const {
    Tensor value = resnet(input, time);
    return attention ? (*attention)(value, context) : value;
  }
};

struct UNetStage {
  std::vector<UNetBlock> blocks;
  std::optional<Conv2d> resize;
};

struct UNet {
  UNetConfig config;
  Conv2d input, output;
  GroupNorm final_norm;
  Linear time_first, time_second;
  UNetBlock mid_first, mid_second;
  std::vector<UNetStage> down, up;
  UNet(UNetConfig config_, const ComponentWeights& weights)
      : config(std::move(config_)), input(model_weights::conv2d(weights, "conv_in", config.in, config.channels.front(), 3, 1, 1)),
        output(model_weights::conv2d(weights, "conv_out", config.channels.front(), config.out, 3, 1, 1)),
        final_norm(model_weights::group_norm(weights, "conv_norm_out", config.channels.front(), config.groups, config.epsilon)),
        time_first(model_weights::linear(weights, "time_embedding.linear_1", config.channels.front(), config.time_width)),
        time_second(model_weights::linear(weights, "time_embedding.linear_2", config.time_width, config.time_width)),
        mid_first(weights, "mid_block", 0, config.channels.back(), config.channels.back(), true, config),
        mid_second(weights, "mid_block", 1, config.channels.back(), config.channels.back(), false, config) {
    int current = config.channels.front();
    std::vector<int> skip{current};
    for (size_t level = 0; level < config.channels.size(); ++level) {
      const std::string prefix = "down_blocks." + std::to_string(level);
      const int width = config.channels[level];
      UNetStage& stage = down.emplace_back();
      for (int layer = 0; layer < config.repeats; ++layer) {
        stage.blocks.emplace_back(weights, prefix, layer, current, width, config.down_attention[level], config);
        current = width;
        skip.push_back(current);
      }
      if (level + 1 < config.channels.size()) {
        stage.resize = model_weights::conv2d(weights, prefix + ".downsamplers.0.conv", width, width, 3, 2, 1);
        skip.push_back(current);
      }
    }
    for (size_t level = 0; level < config.channels.size(); ++level) {
      const std::string prefix = "up_blocks." + std::to_string(level);
      const int width = config.channels[config.channels.size() - 1 - level];
      UNetStage& stage = up.emplace_back();
      for (int layer = 0; layer <= config.repeats; ++layer) {
        stage.blocks.emplace_back(weights, prefix, layer, current + skip.back(), width, config.up_attention[level], config);
        skip.pop_back();
        current = width;
      }
      if (level + 1 < config.channels.size()) stage.resize = model_weights::conv2d(weights, prefix + ".upsamplers.0.conv", width, width, 3, 1, 1);
    }
  }
  Tensor time_embedding(int step, Backend backend, DType dtype) const {
    Tensor features = sinusoidal_embedding(step, config.channels.front(), config.frequency_shift, config.flip, backend, dtype);
    return time_second(silu(time_first(features)));
  }
  Tensor forward(const Tensor& sample, const Tensor& context, int timestep) const {
    const Shape& shape = sample.shape();
    const int factor = 1 << (config.channels.size() - 1);
    if (shape[0] <= 0 || shape[0] != context.shape()[0] || context.shape()[1] <= 0 ||
        shape[2] <= 0 || shape[3] <= 0 || shape[2] % factor || shape[3] % factor)
      throw std::invalid_argument("U-Net latent sizes must match the batch and downsampling factor");
    Tensor time = time_embedding(timestep, sample.backend(), sample.dtype());
    Tensor x = input(sample);
    std::vector<Tensor> skip{x};
    for (const UNetStage& stage : down) {
      for (const UNetBlock& block : stage.blocks) {
        x = block(x, context, time);
        skip.push_back(x);
      }
      if (stage.resize) {
        x = (*stage.resize)(x);
        skip.push_back(x);
      }
    }
    x = mid_second(mid_first(x, context, time), context, time);
    for (const UNetStage& stage : up) {
      for (const UNetBlock& block : stage.blocks) {
        Tensor joined = concat({x, skip.back()}, 1);
        skip.pop_back();
        x = block(joined, context, time);
      }
      if (stage.resize) x = (*stage.resize)(upsample_nearest2d(x));
    }
    return output(silu(final_norm(x)));
  }
};

}  // namespace

InferenceSession load_diffusion_unet(const std::filesystem::path& directory, Backend backend, int device, DType dtype) {
  DeviceScope owner(backend, device);
  const UNetConfig config(Json::read((directory / "config.json").string()));
  std::shared_ptr<UNet> model = std::make_shared<UNet>(config, ComponentWeights(directory, "diffusion_pytorch_model", backend, dtype));
  return InferenceSession(backend, device, {{"sample", {dtype, {-1, config.in, -1, -1}}},
      {"encoder_hidden_states", {dtype, {-1, -1, config.context}}}, {"timestep", {DType::Int32, {}}}},
      {{"sample", {dtype, {-1, config.out, -1, -1}}}}, [model = std::move(model)](const NamedTensors& inputs) {
        return NamedTensors{{"sample", model->forward(inputs.at("sample"), inputs.at("encoder_hidden_states"),
                                                      inputs.at("timestep").int_data().front())}};
      });
}

}  // namespace elman
