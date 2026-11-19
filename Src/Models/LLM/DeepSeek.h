#pragma once

#include "Models/LLM/Decoder.h"
#include "Inference/LLM/Experts.h"
#include "Inference/LLM/ProjectionWeight.h"
#include "Inference/LLM/DevicePlacement.h"
#include "IO/Json.h"

namespace elman {
struct DeepSeekConfig : DecoderConfig {
    int query_rank = 0, plain_dimension = 0, rotary_dimension = 0, value_dimension = 0;
    int expert_intermediate = 0, shared_experts = 0, first_dense_layers = 0, expert_frequency = 1;
    bool attention_bias = false;
    ExpertRoutingOptions routing;
    std::string model_type, rope_type = "default";
    double rope_factor = 1, rope_beta_fast = 32, rope_beta_slow = 1, rope_mscale = 1, rope_mscale_all = 0;
    int rope_original_positions = 4096;
    static DeepSeekConfig from_json(const Json& value);
    bool sparse_layer(int layer) const { return routing.experts > 0 && layer >= first_dense_layers && layer % expert_frequency == 0; }
    double attention_scale() const;
};

// Native DeepSeek V2/V3 inference with compressed latent cache. The optional
// expanded attention route retains HF projection boundaries for diagnostics.
class DeepSeek final : public DecoderModel {
public:
    DeepSeek(const std::string& directory, Backend backend, DType dtype, bool absorbed_attention = true,
             const std::vector<int>& devices = {}, int simulated_devices = 0);
    const DeepSeekConfig& config() const override { return config_; }
    Backend backend() const override { return backend_; }
    DType dtype() const override { return dtype_; }
    int input_device() const override { return placement_.input_device(); }
    const std::vector<int>& device_ordinals() const override { return placement_.ordinals(); }
    int execution_stages() const override { return placement_.stage_count(); }
    bool simulated_device_placement() const override { return placement_.simulated(); }
    size_t activation_transfers() const override { return placement_.activation_transfers(); }
    void synchronize() const override { placement_.synchronize(); }
    size_t parameter_count() const override { return parameter_count_; }
    bool quantized_weights() const override { return quantized_weights_; }
    size_t quantization_workspace_bytes(DecoderLogits logits = DecoderLogits::Last) const override;
    DecoderOutput forward(const Tensor& input_ids, const Tensor& attention_mask = {}, const Tensor& position_ids = {},
        const DecoderCache* past = nullptr, bool capture_hidden = false, DecoderLogits logits = DecoderLogits::All,
        const DecoderTrace& trace = {}) const override;
private:
    struct Expert { ProjectionWeight gate, up, down; };
    struct Layer {
        Tensor input_norm, post_norm, query_norm, latent_norm;
        std::pair<Tensor, Tensor> absorbed_weights;
        ProjectionWeight query, query_a, query_b, latent_a, latent_b, output;
        Tensor query_bias, latent_bias, output_bias, router, correction;
        Expert dense, shared;
        std::vector<Expert> experts;
    };
    DeepSeekConfig config_;
    Backend backend_;
    DType dtype_;
    DecoderPlacement placement_;
    bool absorbed_attention_, quantized_weights_ = false;
    size_t parameter_count_ = 0, layer_workspace_count_ = 0, head_workspace_count_ = 0;
    ProjectionWeight embedding_, head_;
    Tensor norm_;
    std::vector<Layer> layers_;
    std::vector<float> inverse_frequencies_;
    float rotary_magnitude_ = 1;
};
} // namespace elman
