#pragma once
#include "Models/LLM/Decoder.h"
#include <memory>

namespace elman {
class ProcessGroup;
struct DecoderLoadOptions {
    bool fused_pointwise = true, fused_normalization = true, fused_attention_softmax = true;
    bool tiled_attention = true, packed_projections = true, direct_paged_attention = false;
    bool absorbed_attention = true;
    std::vector<int> devices;
    int simulated_devices = 0;
    std::shared_ptr<ProcessGroup> tensor_parallel;
};
DecoderConfig read_decoder_config(const std::string& directory);
std::unique_ptr<DecoderModel> load_decoder(const std::string& directory, Backend backend, DType dtype,
                                         const DecoderLoadOptions& options = {});
} // namespace elman
