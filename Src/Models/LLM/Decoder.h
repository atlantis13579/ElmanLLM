#pragma once

#include "Core/Tensor.h"

namespace elman {
class PagedKVCache;
class PagedKVBatch;

struct DecoderConfig {
    int vocabulary = 0, hidden = 0, intermediate = 0, layers = 0;
    int heads = 0, kv_heads = 0, max_positions = 0;
    float rms_epsilon = 1e-6f, rope_theta = 10000;
    bool tied_embeddings = false;
    int latent_cache_rank = 0, latent_rotary_dimension = 0;
    int attention_query_dimension = 0, attention_value_dimension = 0;
    int routed_expert_count = 0, active_experts = 0, execution_intermediate = 0;
    int head_dim() const { return hidden / heads; }
    size_t cache_elements_per_token() const {
        return latent_cache_rank ? size_t(layers) * (size_t(latent_cache_rank) + latent_rotary_dimension) :
            size_t(2) * layers * kv_heads * head_dim();
    }
};

enum class DecoderLogits { All, Last, None };
struct DecoderLayerCache { Tensor key, value; };
struct DecoderCache {
    std::vector<DecoderLayerCache> layers;
    int length = 0;
};
struct DecoderOutput {
    Tensor logits;
    std::vector<Tensor> hidden_states;
    DecoderCache cache;
};
using DecoderTrace = std::function<void(const std::string&, const Tensor&)>;

// Common inference contract; model implementations own cache layout validation.
class DecoderModel {
public:
    virtual ~DecoderModel() = default;
    virtual const DecoderConfig& config() const = 0;
    virtual Backend backend() const = 0;
    virtual DType dtype() const = 0;
    virtual int input_device() const = 0;
    virtual const std::vector<int>& device_ordinals() const = 0;
    virtual int execution_stages() const { return int(device_ordinals().size()); }
    virtual int tensor_parallel_size() const { return 1; }
    virtual int32_t synchronize_token(int32_t id) const { return id; }
    virtual bool simulated_device_placement() const { return false; }
    virtual size_t activation_transfers() const { return 0; }
    virtual void synchronize() const {}
    virtual size_t parameter_count() const = 0;
    virtual bool quantized_weights() const = 0;
    virtual size_t quantization_workspace_bytes(DecoderLogits logits = DecoderLogits::Last) const = 0;
    virtual bool supports_paged_cache() const { return false; }
    virtual bool fused_pointwise() const { return true; }
    virtual bool fused_normalization() const { return true; }
    virtual bool fused_attention_softmax() const { return true; }
    virtual bool tiled_attention_enabled() const { return true; }
    virtual bool packed_projections() const { return false; }
    virtual bool direct_paged_attention() const { return false; }
    virtual const char* attention_tiling_name() const { return "queries"; }
    // Token IDs may be supplied on the host for validation before the embedding upload.
    virtual DecoderOutput forward(const Tensor& input_ids, const Tensor& attention_mask = {},
        const Tensor& position_ids = {}, const DecoderCache* past = nullptr, bool capture_hidden = false,
        DecoderLogits logits = DecoderLogits::All, const DecoderTrace& trace = {}) const = 0;
    virtual DecoderOutput forward_paged(const Tensor&, PagedKVCache&, const Tensor& = {}, const Tensor& = {},
        bool = false, DecoderLogits = DecoderLogits::All, const DecoderTrace& = {}) const;
    virtual DecoderOutput forward_paged_batch(const Tensor&, PagedKVBatch&, const Tensor& = {}, const Tensor& = {},
        bool = false, DecoderLogits = DecoderLogits::All, const DecoderTrace& = {}) const;
};

// Source compatibility for the original Qwen2 inference API.
using Qwen2Logits = DecoderLogits;
using Qwen2LayerCache = DecoderLayerCache;
using Qwen2Cache = DecoderCache;
using Qwen2Output = DecoderOutput;
} // namespace elman
