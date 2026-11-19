#pragma once

#include "IO/ComponentWeights.h"

namespace elman::flux2 {

struct ReferenceKV {
  Tensor key, value;
};
struct BlockAttention {
  Tensor rotary;
  int text_tokens;
  int reference_tokens = 0;
  const ReferenceKV* reuse = nullptr;
  ReferenceKV* extract = nullptr;
};

struct StreamWeights {
  Tensor qkv, query_norm, key_norm, projection, mlp_in, mlp_out;
};

struct DoubleBlock {
  StreamWeights image, text;
  // Modulation columns are (shift, scale, gate) for attention, then MLP.
  std::pair<Tensor, Tensor> forward(const Tensor& img, const Tensor& txt, const Tensor& img_mod, const Tensor& txt_mod,
                                    const BlockAttention& layout) const;
};

struct SingleBlock {
  Tensor input, output, query_norm, key_norm;
  Tensor forward(const Tensor& hidden, const Tensor& modulation, const BlockAttention& layout) const;
};

enum class Variant { Klein4B, Klein9B, Base4B, Base9B, Klein9BKV, Dev };
struct ReferenceCache {
  std::vector<ReferenceKV> dual, single;
};

struct TransformerDimensions {
  int hidden, context, double_blocks, single_blocks;
  static TransformerDimensions for_variant(Variant variant);
};

class Transformer {
 public:
  Transformer(const io::ComponentWeights& weights, Variant variant);
  Transformer(const io::ComponentWeights& weights, Variant variant, TransformerDimensions dimensions);
  Variant variant() const {
    return variant_;
  }
  // Tables come from rotary_table and can be reused over all denoising steps.
  // Ordinary editing appends references. KV extraction uses [text,ref,image]
  // and fixes reference conditioning at time zero. Reuse omits ref and ref_pe.
  Tensor predict(const Tensor& img, const Tensor& txt, const Tensor& times, const Tensor& img_pe, const Tensor& txt_pe,
                 const Tensor& guidance = {}, const Tensor& ref = {}, const Tensor& ref_pe = {},
                 ReferenceCache* cache = nullptr) const;

 private:
  Variant variant_;
  Tensor img_in_, txt_in_, time_in_, time_out_, guidance_in_, guidance_out_;
  Tensor img_mod_, txt_mod_, single_mod_, final_mod_, final_;
  std::vector<DoubleBlock> dual_;
  std::vector<SingleBlock> single_;
};

}  // namespace elman::flux2
