#pragma once

#include "Core/Tensor.h"
#include "Core/Quantization.h"
#include "Distributed/TensorParallel.h"
#include "Inference/LLM/ProjectionWeight.h"
#include "IO/Json.h"
#include "Models/LLM/Decoder.h"
#include "Inference/LLM/DevicePlacement.h"

namespace elman {
class PagedKVCache;
class PagedKVAppend;
class PagedKVBatch;
class PagedAttentionMap;
struct PagedKVLayer;
// Keep common prefill chunks in one GEMM row group. Smaller query tiles can
// change long-key BF16 reduction rounding relative to the eager reference.
inline constexpr int kQwenAttentionQueryTile = 128;

struct Qwen2Config : DecoderConfig {
    static Qwen2Config from_json(const Json& value);
};

// Eager inference with immutable contiguous caches or transactional paged
// storage. Direct paged attention optionally reads physical slots without K/V gathers.
class Qwen2 : public DecoderModel {
public:
    Qwen2(const std::string& model_directory, Backend backend, DType dtype, bool fused_pointwise = true,
           bool fused_normalization = true, bool fused_attention_softmax = true, bool tiled_attention = true,
           bool packed_projections = true, bool direct_paged_attention = false, const std::vector<int>& devices = {},
           int simulated_devices = 0, std::shared_ptr<ProcessGroup> tensor_parallel = {});
    const Qwen2Config& config() const { return config_; }
    bool supports_paged_cache() const override { return !placement_.distributed() && tensor_parallel_size() == 1; }
    int input_device() const override { return placement_.input_device(); }
    const std::vector<int>& device_ordinals() const override { return placement_.ordinals(); }
    int execution_stages() const override { return placement_.stage_count(); }
    int tensor_parallel_size() const override { return parallel_ ? parallel_->size() : 1; }
    int32_t synchronize_token(int32_t id) const override;
    bool simulated_device_placement() const override { return placement_.simulated(); }
    size_t activation_transfers() const override { return placement_.activation_transfers(); }
    void synchronize() const override { placement_.synchronize(); }
    Backend backend() const { return backend_; }
    DType dtype() const { return dtype_; }
    bool fused_pointwise() const { return fused_pointwise_; }
    bool fused_normalization() const { return fused_normalization_; }
    bool fused_attention_softmax() const { return fused_attention_softmax_; }
    bool tiled_attention_enabled() const { return tiled_attention_; }
    bool packed_projections() const { return packed_projections_; }
    bool direct_paged_attention() const { return direct_paged_attention_; }
    const char* attention_tiling_name() const { return !tiled_attention_ ? "none" : backend_ == Backend::CPU ? "heads" : "queries"; }
    size_t parameter_count() const { return parameter_count_; }
    bool quantized_weights() const { return quantized_weights_; }
    size_t quantization_workspace_bytes(Qwen2Logits logits = Qwen2Logits::Last) const;
    // IDs/positions: Int32 or Int64 [batch,sequence]. Mask: 0/1 [batch,total].
    // Default positions count non-padding tokens, assigning zero to padding.
    // Explicit positions override this rule. Defined logits are Float32.
    // Last projects only the final position; None skips the final norm and
    // head unless capture_hidden requires the normalized hidden state.
    Qwen2Output forward(const Tensor& input_ids, const Tensor& attention_mask = {},
                        const Tensor& position_ids = {}, const Qwen2Cache* past = nullptr,
                        bool capture_hidden = false, Qwen2Logits logits = Qwen2Logits::All,
                        const std::function<void(const std::string&, const Tensor&)>& trace = {}) const;
    // The cache belongs to this model/session. Returned cache tensors are
    // diagnostic snapshots only when capture_hidden is enabled; length is always
    // populated. Failure leaves the committed request prefix unchanged.
    Qwen2Output forward_paged(const Tensor& input_ids, PagedKVCache& cache,
                              const Tensor& attention_mask = {}, const Tensor& position_ids = {},
                              bool capture_hidden = false, Qwen2Logits logits = Qwen2Logits::All,
                              const std::function<void(const std::string&, const Tensor&)>& trace = {}) const;
    // Execute distinct requests in one model batch, with the same query length
    // and possibly different committed history lengths. Structural left padding
    // is always masked. Default positions count each row's real tokens. Explicit
    // masks may hide real tokens but cannot expose structural padding. Returned
    // diagnostic cache tensors are left-padded; request caches remain unpadded.
    // Every participating append commits together, or every prefix is preserved.
    Qwen2Output forward_paged_batch(const Tensor& input_ids, PagedKVBatch& cache,
                                    const Tensor& attention_mask = {}, const Tensor& position_ids = {},
                                    bool capture_hidden = false, Qwen2Logits logits = Qwen2Logits::All,
                                    const std::function<void(const std::string&, const Tensor&)>& trace = {}) const;

private:
    friend class Qwen2Graph;
    friend class Qwen2PagedGraph;
    struct GraphInputs { Tensor indices, cosine, sine, padding, causal; };
    struct PagedGraphInputs {
        const PagedAttentionMap& map;
        const Tensor& destinations;
        const std::vector<PagedKVLayer>& storage;
    };
    std::pair<std::vector<float>, std::vector<float>> rotary_values(const std::vector<int32_t>& positions) const;
    Qwen2Output forward_impl(const Tensor& input_ids, const Tensor& attention_mask,
                             const Tensor& position_ids, const Qwen2Cache* past, PagedKVAppend* append,
                             bool capture_hidden, Qwen2Logits logits,
                             const std::function<void(const std::string&, const Tensor&)>& trace, GraphInputs* graph_inputs = nullptr,
                             const PagedGraphInputs* graph_paged = nullptr) const;
    using Weight = ProjectionWeight;
    struct Layer {
        Tensor input_norm, post_norm;
        Weight query, key, value, output, gate, up, down, qkv, gate_up;
        Tensor query_bias, key_bias, value_bias, qkv_bias;
    };
    Qwen2Config config_;
    Backend backend_;
    DType dtype_;
    DecoderPlacement placement_;
    std::shared_ptr<ProcessGroup> parallel_;
    TensorParallelHeads parallel_heads_;
    int local_intermediate_;
    bool fused_pointwise_;
    bool fused_normalization_;
    bool fused_attention_softmax_;
    bool tiled_attention_;
    bool packed_projections_;
    bool direct_paged_attention_;
    size_t parameter_count_ = 0;
    Weight embedding_, lm_head_;
    Tensor norm_;
    bool quantized_weights_ = false;
    size_t quantized_layer_count_ = 0, quantized_head_count_ = 0;
    std::vector<Layer> layers_;
    std::vector<float> inverse_frequencies_;
};

} // namespace elman
