#pragma once

#include "Core/CommandGraph.h"
#include "Inference/LLM/PagedKVCache.h"
#include "Models/LLM/Qwen2.h"

namespace elman {

// A fixed batch-width and context bucket for one worker's paged cache pool.
// Tokens, positions, logical page slots and write destinations change at replay.
class Qwen2PagedGraph {
 public:
  Qwen2PagedGraph(const Qwen2& model, PagedKVBatch& batch, int capacity,
                  const std::vector<int32_t>& tokens);
  Tensor decode(PagedKVBatch& batch, const std::vector<int32_t>& tokens);
  const Tensor& logits() const { return logits_; }
  bool matches(int batch, int capacity) const { return batch_ == batch && capacity_ == capacity; }
  size_t commands() const { return graph_.command_count(); }
  size_t nodes() const { return graph_.node_count(); }
  size_t retained_bytes() const { return graph_.retained_bytes(); }
  static int context_bucket(int required, int maximum);
  static size_t memory_bound(const Qwen2& model, int batch, int capacity);

 private:
  const Qwen2& model_;
  const int batch_, capacity_;
  std::unique_ptr<PagedAttentionMap> map_;
  Qwen2::GraphInputs inputs_;
  Tensor destinations_, logits_;
  std::vector<PagedKVLayer> storage_;
  CommandGraph graph_;
};

}  // namespace elman
