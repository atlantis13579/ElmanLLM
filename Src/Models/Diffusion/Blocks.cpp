#include "Models/Diffusion/Blocks.h"

#include <stdexcept>
#include <utility>

#include "Operators/inference_ops.h"

namespace elman::diffusion {

ResnetBlock::ResnetBlock(const io::ComponentWeights& weights, const std::string& prefix, int input, int output,
                         int groups, double epsilon, int time_width, const char* shortcut_name)
    : first_norm(model_weights::group_norm(weights, prefix + ".norm1", input, groups, epsilon)),
      second_norm(model_weights::group_norm(weights, prefix + ".norm2", output, groups, epsilon)),
      first(model_weights::conv2d(weights, prefix + ".conv1", input, output, 3, 1, 1)),
      second(model_weights::conv2d(weights, prefix + ".conv2", output, output, 3, 1, 1)) {
  if (time_width) time = model_weights::linear(weights, prefix + ".time_emb_proj", time_width, output);
  if (input != output || weights.contains(prefix + shortcut_name + ".weight"))
    shortcut = model_weights::conv2d(weights, prefix + shortcut_name, input, output, 1);
}

Tensor ResnetBlock::operator()(const Tensor& input, const Tensor& time_embedding) const {
  auto value = first(silu(first_norm(input)));
  if (time) {
    auto projected = (*time)(silu(time_embedding));
    value = add(value, reshape(projected, {projected.shape()[0], projected.shape()[1], 1, 1}));
  }
  value = second(silu(second_norm(value)));
  return add(shortcut ? (*shortcut)(input) : input, value);
}

namespace {

inference::Attention transformer_attention(const io::ComponentWeights& weights, const std::string& prefix, int width, int context, int heads) {
  return inference::Attention(model_weights::linear(weights, prefix + ".to_q", width, width, false),
                   model_weights::linear(weights, prefix + ".to_k", context, width, false),
                   model_weights::linear(weights, prefix + ".to_v", context, width, false),
                   model_weights::linear(weights, prefix + ".to_out.0", width, width), heads);
}

}  // namespace

SpatialTransformer::SpatialTransformer(const io::ComponentWeights& weights, const std::string& prefix, int width,
                                       int context, int heads, int groups)
    : group_norm(model_weights::group_norm(weights, prefix + ".norm", width, groups, 1e-6)),
      project_in(model_weights::conv2d(weights, prefix + ".proj_in", width, width, 1)), project_out(model_weights::conv2d(weights, prefix + ".proj_out", width, width, 1)),
      self_norm(model_weights::layer_norm(weights, prefix + ".transformer_blocks.0.norm1", width, 1e-5)),
      cross_norm(model_weights::layer_norm(weights, prefix + ".transformer_blocks.0.norm2", width, 1e-5)),
      ff_norm(model_weights::layer_norm(weights, prefix + ".transformer_blocks.0.norm3", width, 1e-5)),
      self_attention(transformer_attention(weights, prefix + ".transformer_blocks.0.attn1", width, width, heads)),
      cross_attention(transformer_attention(weights, prefix + ".transformer_blocks.0.attn2", width, context, heads)),
      ff_gate(model_weights::linear(weights, prefix + ".transformer_blocks.0.ff.net.0.proj", width, width * 8)),
      ff_out(model_weights::linear(weights, prefix + ".transformer_blocks.0.ff.net.2", width * 4, width)), widened(width * 4) {}

Tensor SpatialTransformer::operator()(const Tensor& input, const Tensor& context) const {
  auto x = image_tokens(project_in(group_norm(input)));
  const auto normalized = self_norm(x);
  x = add(x, self_attention(normalized, normalized));
  x = add(x, cross_attention(cross_norm(x), context));
  const auto projected = ff_gate(ff_norm(x));
  const auto value = slice(projected, -1, 0, widened);
  const auto gate = slice(projected, -1, widened, 2 * widened);
  x = add(x, ff_out(mul(value, exact_gelu(gate))));
  return add(input, project_out(token_image(x, input.shape()[2], input.shape()[3])));
}

}  // namespace elman::diffusion
