#pragma once

#include "Models/LLM/Decoder.h"
#include "IO/Gguf.h"

namespace elman {

struct Qwen35Config : DecoderConfig {
    int rotary_dimensions = 0, convolution_kernel = 0, key_heads = 0, value_heads = 0;
    int key_width = 0, value_width = 0;
    std::vector<bool> recurrent;
    static Qwen35Config from_gguf(const GgufFile& file);
};

// Native Qwen3.5 hybrid decoder, including the IndexTranslate text model.
// Recurrent cache layers store convolution history in key and DeltaNet state in value.
class Qwen35 final : public DecoderModel {
public:
    Qwen35(const std::filesystem::path& model, Backend backend, DType dtype);
    ~Qwen35();
    const Qwen35Config& config() const override;
    Backend backend() const override;
    DType dtype() const override;
    int input_device() const override;
    const std::vector<int>& device_ordinals() const override;
    size_t parameter_count() const override;
    bool quantized_weights() const override;
    size_t quantization_workspace_bytes(DecoderLogits logits = DecoderLogits::Last) const override;
    DecoderOutput forward(const Tensor& input_ids, const Tensor& attention_mask = {}, const Tensor& position_ids = {},
        const DecoderCache* past = nullptr, bool capture_hidden = false, DecoderLogits logits = DecoderLogits::All,
        const DecoderTrace& trace = {}) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elman
