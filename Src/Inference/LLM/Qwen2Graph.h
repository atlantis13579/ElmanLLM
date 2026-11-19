#pragma once
#include "Models/LLM/Qwen2.h"
#include "Core/CommandGraph.h"

namespace elman {
// One request, one CUDA device and one fixed context capacity. Prefill uses the
// ordinary model. Every decode replay includes embedding, all decoder layers,
// final norm/head and cache updates. Returned logits alias reusable graph storage.
class Qwen2Graph {
public:
    Qwen2Graph(const Qwen2& model, int capacity, const DecoderCache& prefix);
    Tensor decode(int32_t token);
    DecoderCache cache_snapshot() const;
    const std::vector<Tensor>& hidden_states() const { return hidden_states_; }
    int length() const { return length_; }
    size_t commands() const { return graph_.command_count(); }
    size_t nodes() const { return graph_.node_count(); }
    size_t retained_bytes() const { return graph_.retained_bytes(); }
    void synchronize() const { graph_.synchronize(); }
    static size_t memory_bound(const DecoderConfig& config, DType dtype, int capacity, size_t weight_workspace);
private:
    const Qwen2& model_;
    int capacity_, length_;
    Qwen2::GraphInputs inputs_;
    DecoderCache storage_;
    DecoderCache output_cache_;
    std::vector<Tensor> hidden_states_;
    Tensor cache_index_, logits_;
    CommandGraph graph_;
};
} // namespace elman
