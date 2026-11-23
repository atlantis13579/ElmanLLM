#include "Models/Diffusion/Flux2Vae.h"

#include <memory>

#include "Models/Diffusion/Blocks.h"
#include "Operators/flux2_ops.h"
#include "Operators/inference_ops.h"

namespace elman::flux2 {
namespace {
using diffusion::ResnetBlock;
using inference::Conv2d;
using inference::GroupNorm;
using inference::Linear;
using io::ComponentWeights;

struct Middle {
  ResnetBlock first, second;
  GroupNorm norm;
  Linear query, key, value, output;

  Middle(const ComponentWeights& w, const std::string& p, bool diffusers)
      : first(w, p + (diffusers ? ".mid_block.resnets.0" : ".mid.block_1"), 512, 512, 32, 1e-6),
        second(w, p + (diffusers ? ".mid_block.resnets.1" : ".mid.block_2"), 512, 512, 32, 1e-6),
        norm(model_weights::group_norm(w, p + (diffusers ? ".mid_block.attentions.0.group_norm" : ".mid.attn_1.norm"),
                                       512, 32, 1e-6)),
        query(projection(w, p, diffusers, ".to_q", ".q")),
        key(projection(w, p, diffusers, ".to_k", ".k")),
        value(projection(w, p, diffusers, ".to_v", ".v")),
        output(projection(w, p, diffusers, ".to_out.0", ".proj_out")) {
  }

  static Linear projection(const ComponentWeights& w, const std::string& p, bool diffusers,
                           const char* diffusers_name, const char* original_name) {
    const std::string prefix = diffusers ? p + ".mid_block.attentions.0" + diffusers_name : p + ".mid.attn_1" + original_name;
    return model_weights::linear(w, prefix, 512, 512, true, true);
  }

  Tensor operator()(const Tensor& input) const {
    Tensor x = first(input), h = image_tokens(norm(x));
    Tensor a = scaled_dot_product_attention(unsqueeze(query(h), 1), unsqueeze(key(h), 1), unsqueeze(value(h), 1), false);
    return second(add(x, token_image(output(squeeze(a, 1)), x.shape()[2], x.shape()[3])));
  }
};

struct Stage {
  std::vector<ResnetBlock> blocks;
  std::optional<Conv2d> resize;
};

Tensor pad_right_bottom(const Tensor& x) {
  const Shape& s = x.shape();
  Tensor column = Tensor::zeros({s[0], s[1], s[2], 1}, x.backend(), false, x.dtype());
  Tensor row = Tensor::zeros({s[0], s[1], 1, s[3] + 1}, x.backend(), false, x.dtype());
  return concat({concat({x, column}, 3), row}, 2);
}

struct Vae {
  bool encode;
  Middle mid;
  GroupNorm norm;
  Conv2d input, output, quant;
  Tensor mean, variance;
  std::vector<Stage> stages;

  Vae(const ComponentWeights& w, const std::string& p, bool encoder, bool diffusers)
      : encode(encoder),
        mid(w, p, diffusers),
        norm(model_weights::group_norm(w, p + (diffusers ? ".conv_norm_out" : ".norm_out"),
                                       encode ? 512 : 128, 32, 1e-6)),
        input(model_weights::conv2d(w, p + ".conv_in", encode ? 3 : 32, encode ? 128 : 512, 3, 1, 1)),
        output(model_weights::conv2d(w, p + ".conv_out", encode ? 512 : 128, encode ? 64 : 3, 3, 1, 1)),
        quant(encode ? model_weights::conv2d(w, diffusers ? "quant_conv" : p + ".quant_conv", 64, 64, 1)
                     : model_weights::conv2d(w, diffusers ? "post_quant_conv" : p + ".post_quant_conv", 32, 32, 1)),
        mean(w.get("bn.running_mean", {128}, DType::Float32)),
        variance(w.get("bn.running_var", {128}, DType::Float32)) {
    constexpr int channels[] = {128, 256, 512, 512};
    int width = encode ? 128 : 512;
    for (int step = 0; step < 4; ++step) {
      const int level = encode ? step : 3 - step, next = channels[level];
      std::string prefix = diffusers ? p + (encode ? ".down_blocks." : ".up_blocks.") + std::to_string(step)
                              : p + (encode ? ".down." : ".up.") + std::to_string(level);
      flux2::Stage& stage = stages.emplace_back();
      for (int j = 0; j < (encode ? 2 : 3); ++j) {
        stage.blocks.emplace_back(w, prefix + (diffusers ? ".resnets." : ".block.") + std::to_string(j),
                                  width, next, 32, 1e-6, 0, diffusers ? ".conv_shortcut" : ".nin_shortcut");
        width = next;
      }
      if (step < 3) {
        const char* const resize_name = diffusers ? (encode ? ".downsamplers.0.conv" : ".upsamplers.0.conv")
                                           : (encode ? ".downsample.conv" : ".upsample.conv");
        stage.resize = model_weights::conv2d(w, prefix + resize_name, width, width, 3, encode ? 2 : 1, encode ? 0 : 1);
      }
    }
  }

  Tensor forward(const Tensor& data) const {
    const Shape& s = data.shape();
    if (s[0] < 1 || s[2] < 1 || s[3] < 1 || (encode && (s[2] % 16 || s[3] % 16))) {
      throw std::invalid_argument("FLUX.2 reference image sides must be multiples of 16 and latents must be nonempty");
    }
    Tensor x = encode ? input(data) : mid(input(quant(unpack_latents(data, mean, variance))));
    for (const flux2::Stage& stage : stages) {
      for (const diffusion::ResnetBlock& block : stage.blocks) {
        x = block(x);
      }
      if (stage.resize) {
        x = (*stage.resize)(encode ? pad_right_bottom(x) : upsample_nearest2d(x));
      }
    }
    if (encode) {
      x = mid(x);
    }
    x = output(silu(norm(x)));
    return encode ? pack_latents(quant(x), mean, variance) : x;
  }
};

InferenceSession session(const ComponentWeights& w, bool encode) {
  const bool diffusers = w.contains("decoder.conv_norm_out.weight");
  std::shared_ptr<flux2::Vae> model = std::make_shared<Vae>(w, encode ? "encoder" : "decoder", encode, diffusers);
  const Tensor& weight = model->input.weight;
  const std::string input = encode ? "image" : "latent", output = encode ? "latent" : "image";
  return InferenceSession(
      weight.backend(), weight.device(), {{input, {weight.dtype(), {-1, encode ? 3 : 128, -1, -1}}}},
      {{output, {weight.dtype(), {-1, encode ? 128 : 3, -1, -1}}}}, [model, input, output](const NamedTensors& args) {
        return NamedTensors{{output, model->forward(args.at(input))}};
      });
}
}  // namespace

InferenceSession load_vae_encoder(const ComponentWeights& w) {
  return session(w, true);
}
InferenceSession load_vae_decoder(const ComponentWeights& w) {
  return session(w, false);
}
}  // namespace elman::flux2
