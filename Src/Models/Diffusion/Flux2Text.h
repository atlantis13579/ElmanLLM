#pragma once

#include "IO/Safetensors.h"

namespace elman::flux2 {

// Consume token IDs and their 0/1 padding mask from the checkpoint processor.
// The encoder streams weights and retains only the three requested HF states.
class TextEncoder {
 public:
  explicit TextEncoder(const std::filesystem::path& directory, DType dtype);
  Tensor encode(const Tensor& input_ids, const Tensor& attention_mask) const;

 private:
  SafetensorsWeights weights_;
  DType dtype_;
  int width_, intermediate_, heads_, kv_heads_, dimension_, vocab_;
  bool qwen_;
  double epsilon_, theta_;
  std::string prefix_;
};
}  // namespace elman::flux2
