#include "Models/Diffusion/Flux2Transformer.h"

#include <stdexcept>
#include <tuple>

#include "Core/Device.h"
#include "Operators/flux2_ops.h"

namespace elman::flux2 {
namespace {
Tensor project(const Tensor& input, const Tensor& weight) {
  return linear(input, weight.to(input.backend(), input.device()));
}

Tensor tokens(const Tensor& x, int begin, int end) {
  return slice(x, 2, begin, end);
}

Tensor joint_attention(const Tensor& q, Tensor k, Tensor v, const BlockAttention& layout) {
  const int text = layout.text_tokens, refs = layout.reference_tokens, length = q.shape()[2];
  if (text < 0 || text > length || refs < 0 || refs > length - text || (layout.reuse && (refs || layout.extract)) ||
      (layout.extract && !refs)) {
    throw std::invalid_argument("Invalid FLUX.2 reference attention layout");
  }
  Tensor output;
  if (layout.reuse) {
    auto insert_refs = [&](const Tensor& x, const Tensor& cached) {
      return concat({tokens(x, 0, text), cached, tokens(x, text, length)}, 2);
    };
    k = insert_refs(k, layout.reuse->key);
    v = insert_refs(v, layout.reuse->value);
  }
  if (!refs) {
    output = attention(q, k, v);
  } else {
    const int end = text + refs;
    auto kr = tokens(k, text, end), vr = tokens(v, text, end);
    auto live = attention(concat({tokens(q, 0, text), tokens(q, end, length)}, 2), k, v);
    auto fixed = attention(tokens(q, text, end), kr, vr);
    output = concat({tokens(live, 0, text), fixed, tokens(live, text, length - refs)}, 2);
    if (layout.extract) {
      *layout.extract = {kr, vr};
    }
  }
  return reshape(transpose(output, 1, 2), {q.shape()[0], length, q.shape()[1] * 128});
}

struct Qkv {
  Tensor q, k, v;
};
Qkv prepare(const Tensor& projected, int width, const Tensor& q_norm, const Tensor& k_norm) {
  const auto& s = projected.shape();
  if (s.size() != 3 || width <= 0 || width % 128 || s[2] / 3 < width) {
    throw std::invalid_argument("Invalid projected FLUX.2 attention width");
  }
  auto heads = [&](int part) {
    auto channels = slice(projected, -1, part * width, (part + 1) * width);
    return transpose(reshape(channels, {s[0], s[1], width / 128, 128}), 1, 2);
  };
  return {rms_norm(heads(0), q_norm.to(projected.backend(), projected.device()), 1e-6, RmsNormRounding::Normalized),
          rms_norm(heads(1), k_norm.to(projected.backend(), projected.device()), 1e-6, RmsNormRounding::Normalized),
          heads(2)};
}

Tensor gated_mlp(const Tensor& channels) {
  const int size = channels.shape().back();
  if (!size || size % 2) {
    throw std::invalid_argument("Invalid FLUX.2 SwiGLU width");
  }
  return silu_mul(slice(channels, -1, 0, size / 2), slice(channels, -1, size / 2, size));
}

std::vector<Tensor> chunks(const Tensor& modulation, int count, int width) {
  if (modulation.shape().size() != 3 || width <= 0 || int64_t(width) * count != modulation.shape()[2]) {
    throw std::invalid_argument("Invalid FLUX.2 modulation width");
  }
  return split(modulation, width, -1);
}

Tensor finish_stream(const Tensor& x, const Tensor& attended, const std::vector<Tensor>& mods, const StreamWeights& w) {
  auto h = add(x, mul(mods[2], project(attended, w.projection)));
  auto mlp = gated_mlp(project(modulate(h, mods[3], mods[4]), w.mlp_in));
  return add(h, mul(mods[5], project(mlp, w.mlp_out)));
}

Tensor embed(const Tensor& values, const Tensor& in, const Tensor& out) {
  return project(silu(project(timestep_embedding(values), in)), out);
}

Tensor blend(const Tensor& live, const Tensor& fixed, int text, int refs, int img) {
  auto segment = [](const Tensor& x, int length) { return expand(x, {x.shape()[0], length, x.shape()[2]}); };
  if (!text) {
    return concat({segment(fixed, refs), segment(live, img)}, 1);
  }
  return concat({segment(live, text), segment(fixed, refs), segment(live, img)}, 1);
}
}  // namespace

std::pair<Tensor, Tensor> DoubleBlock::forward(const Tensor& img, const Tensor& txt, const Tensor& img_mod,
                                               const Tensor& txt_mod, const BlockAttention& layout) const {
  const int width = img.shape().at(2), text_len = txt.shape().at(1);
  if (width != txt.shape().at(2) || text_len != layout.text_tokens) {
    throw std::invalid_argument("FLUX.2 stream dimensions differ");
  }
  auto im = chunks(img_mod, 6, width), tm = chunks(txt_mod, 6, width);
  auto iq = prepare(project(modulate(img, im[0], im[1]), image.qkv), width, image.query_norm, image.key_norm);
  auto tq = prepare(project(modulate(txt, tm[0], tm[1]), text.qkv), width, text.query_norm, text.key_norm);
  auto rotated = [&](const Tensor& left, const Tensor& right) {
    return apply_rotary(concat({left, right}, 2), layout.rotary);
  };
  auto attended = joint_attention(rotated(tq.q, iq.q), rotated(tq.k, iq.k), concat({tq.v, iq.v}, 2), layout);
  return {finish_stream(img, slice(attended, 1, text_len, attended.shape()[1]), im, image),
          finish_stream(txt, slice(attended, 1, 0, text_len), tm, text)};
}

Tensor SingleBlock::forward(const Tensor& hidden, const Tensor& modulation, const BlockAttention& layout) const {
  const int width = hidden.shape().at(2);
  auto mods = chunks(modulation, 3, width);
  auto packed = project(modulate(hidden, mods[0], mods[1]), input);
  auto qkv = prepare(packed, width, query_norm, key_norm);
  auto attended =
      joint_attention(apply_rotary(qkv.q, layout.rotary), apply_rotary(qkv.k, layout.rotary), qkv.v, layout);
  auto mlp = gated_mlp(slice(packed, -1, 3 * width, packed.shape()[2]));
  return add(hidden, mul(mods[2], project(concat({attended, mlp}, -1), output)));
}

TransformerDimensions TransformerDimensions::for_variant(Variant variant) {
  switch (variant) {
    case Variant::Dev:
      return {6144, 15360, 8, 48};
    case Variant::Klein4B:
    case Variant::Base4B:
      return {3072, 7680, 5, 20};
    case Variant::Klein9B:
    case Variant::Base9B:
    case Variant::Klein9BKV:
      return {4096, 12288, 8, 24};
  }
  throw std::invalid_argument("Unknown FLUX.2 variant");
}

Transformer::Transformer(const io::ComponentWeights& w, Variant variant)
    : Transformer(w, variant, TransformerDimensions::for_variant(variant)) {
}

Transformer::Transformer(const io::ComponentWeights& w, Variant variant, TransformerDimensions dims)
    : variant_(variant) {
  const int dim = dims.hidden;
  if (dim <= 0 || dim % 128 || dim > INT32_MAX / 9 || dims.context <= 0 || dims.double_blocks <= 0 ||
      dims.single_blocks <= 0) {
    throw std::invalid_argument("Invalid FLUX.2 transformer dimensions");
  }
  auto weight = [&](const std::string& prefix, int out, int in) { return w.get(prefix + ".weight", {out, in}); };
  img_in_ = weight("img_in", dim, 128);
  txt_in_ = weight("txt_in", dim, dims.context);
  time_in_ = weight("time_in.in_layer", dim, 256);
  time_out_ = weight("time_in.out_layer", dim, dim);
  if (variant == Variant::Dev) {
    guidance_in_ = weight("guidance_in.in_layer", dim, 256);
    guidance_out_ = weight("guidance_in.out_layer", dim, dim);
  }
  img_mod_ = weight("double_stream_modulation_img.lin", dim * 6, dim);
  txt_mod_ = weight("double_stream_modulation_txt.lin", dim * 6, dim);
  single_mod_ = weight("single_stream_modulation.lin", dim * 3, dim);
  final_mod_ = weight("final_layer.adaLN_modulation.1", dim * 2, dim);
  final_ = weight("final_layer.linear", 128, dim);
  for (int i = 0; i < dims.double_blocks; ++i) {
    const auto prefix = "double_blocks." + std::to_string(i) + ".";
    auto stream = [&](const std::string& name) -> StreamWeights {
      const auto attn = prefix + name + "_attn.", mlp = prefix + name + "_mlp.";
      return {weight(attn + "qkv", 3 * dim, dim),
              w.get(attn + "norm.query_norm.scale", {128}),
              w.get(attn + "norm.key_norm.scale", {128}),
              weight(attn + "proj", dim, dim),
              weight(mlp + "0", 6 * dim, dim),
              weight(mlp + "2", dim, 3 * dim)};
    };
    dual_.push_back({stream("img"), stream("txt")});
  }
  for (int i = 0; i < dims.single_blocks; ++i) {
    const auto prefix = "single_blocks." + std::to_string(i) + ".";
    single_.push_back({weight(prefix + "linear1", 9 * dim, dim), weight(prefix + "linear2", dim, 4 * dim),
                       w.get(prefix + "norm.query_norm.scale", {128}), w.get(prefix + "norm.key_norm.scale", {128})});
  }
}

Tensor Transformer::predict(const Tensor& img, const Tensor& txt, const Tensor& times, const Tensor& img_pe,
                            const Tensor& txt_pe, const Tensor& guidance, const Tensor& ref, const Tensor& ref_pe,
                            ReferenceCache* cache) const {
  const int batch = img.shape().at(0), target = img.shape().at(1), text = txt.shape().at(1);
  auto check_table = [&](const Tensor& stream, const Tensor& table) {
    const auto& s = stream.shape();
    const auto& t = table.shape();
    if (s.size() != 3 || s[0] != batch || s[1] < 1 || t.size() != 3 || (t[0] != 1 && t[0] != batch) || t[1] != s[1] ||
        t[2] != 128 || table.dtype() != DType::Float32) {
      throw std::invalid_argument("FLUX.2 position table must match its token stream");
    }
  };
  auto join_tables = [&](Tensor left, Tensor right) {
    if (left.shape()[0] != right.shape()[0]) {
      if (left.shape()[0] == 1) {
        left = expand(left, {batch, left.shape()[1], 128});
      } else {
        right = expand(right, {batch, right.shape()[1], 128});
      }
    }
    return concat({left, right}, 1);
  };
  check_table(img, img_pe);
  check_table(txt, txt_pe);
  if (times.shape() != Shape{batch} || txt.shape()[0] != batch || ref.defined() != ref_pe.defined() ||
      (cache && variant_ != Variant::Klein9BKV) ||
      (guidance_in_.defined() && (!guidance.defined() || guidance.shape() != times.shape()))) {
    throw std::invalid_argument("Incompatible FLUX.2 denoiser conditioning");
  }
  DeviceScope owner(img.backend(), img.device());
  const bool extract = cache && ref.defined(), reuse = cache && !extract;
  if ((reuse && (cache->dual.size() != dual_.size() || cache->single.size() != single_.size())) ||
      (variant_ == Variant::Klein9BKV && ref.defined() && !cache)) {
    throw std::invalid_argument("Invalid FLUX.2 per-block reference cache");
  }
  ReferenceCache fresh;
  if (extract) {
    fresh.dual.resize(dual_.size());
    fresh.single.resize(single_.size());
  }
  auto condition = [&](const Tensor& t) {
    auto vec = embed(t, time_in_, time_out_);
    if (guidance_in_.defined()) {
      vec = add(vec, embed(guidance, guidance_in_, guidance_out_));
    }
    return unsqueeze(silu(vec), 1);
  };
  auto cond = condition(times);
  auto im = project(cond, img_mod_), tm = project(cond, txt_mod_), sm = project(cond, single_mod_);
  auto x = img, pe = img_pe;
  const int refs = ref.defined() ? ref.shape().at(1) : 0;
  if (ref.defined()) {
    check_table(ref, ref_pe);
    x = extract ? concat({ref, img}, 1) : concat({img, ref}, 1);
    pe = extract ? join_tables(ref_pe, img_pe) : join_tables(img_pe, ref_pe);
    if (extract) {
      auto fixed = condition(Tensor::zeros({batch}, img.backend(), false, times.dtype()));
      im = blend(im, project(fixed, img_mod_), 0, refs, target);
      sm = blend(sm, project(fixed, single_mod_), text, refs, target);
    }
  }
  x = project(x, img_in_);
  auto ctx = project(txt, txt_in_);
  BlockAttention layout{join_tables(txt_pe, pe), text, extract ? refs : 0};
  for (size_t i = 0; i < dual_.size(); ++i) {
    layout.extract = extract ? &fresh.dual[i] : nullptr;
    layout.reuse = reuse ? &cache->dual[i] : nullptr;
    std::tie(x, ctx) = dual_[i].forward(x, ctx, im, tm, layout);
  }
  x = concat({ctx, x}, 1);
  for (size_t i = 0; i < single_.size(); ++i) {
    layout.extract = extract ? &fresh.single[i] : nullptr;
    layout.reuse = reuse ? &cache->single[i] : nullptr;
    x = single_[i].forward(x, sm, layout);
  }
  const int begin = text + (extract ? refs : 0);
  auto final_mods = chunks(project(cond, final_mod_), 2, x.shape()[2]);
  auto velocity = project(modulate(slice(x, 1, begin, begin + target), final_mods[0], final_mods[1]), final_);
  if (extract) {
    *cache = std::move(fresh);
  }
  return velocity;
}
}  // namespace elman::flux2
