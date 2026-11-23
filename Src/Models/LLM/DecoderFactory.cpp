#include "Models/LLM/DecoderFactory.h"
#include "Models/LLM/Qwen2.h"
#include "Models/LLM/DeepSeek.h"
#include <filesystem>
#include <stdexcept>

namespace elman {
DecoderConfig read_decoder_config(const std::string& directory) {
    Json value = Json::read((std::filesystem::path(directory) / "config.json").string());
    const std::string type = value.at("model_type").string();
    if (type == "qwen2") return Qwen2Config::from_json(value);
    if (type == "deepseek_v2" || type == "deepseek_v3") return DeepSeekConfig::from_json(value);
    throw std::invalid_argument("Unsupported decoder model_type: " + type);
}
std::unique_ptr<DecoderModel> load_decoder(const std::string& directory, Backend backend, DType dtype,
                                         const DecoderLoadOptions& options) {
    Json value = Json::read((std::filesystem::path(directory) / "config.json").string());
    const std::string type = value.at("model_type").string();
    if (type == "qwen2") return std::make_unique<Qwen2>(directory, backend, dtype, options.fused_pointwise,
        options.fused_normalization, options.fused_attention_softmax, options.tiled_attention,
        options.packed_projections, options.direct_paged_attention, options.devices, options.simulated_devices, options.tensor_parallel);
    if (type == "deepseek_v2" || type == "deepseek_v3") {
        if (options.tensor_parallel) throw std::invalid_argument("Tensor-parallel decoding requires dense Qwen2 weights");
        if (options.direct_paged_attention) throw std::invalid_argument("DeepSeek uses a compressed latent cache; direct paged attention is a Qwen2 option");
        return std::make_unique<DeepSeek>(directory, backend, dtype, options.absorbed_attention, options.devices, options.simulated_devices);
    }
    throw std::invalid_argument("Unsupported decoder model_type: " + type);
}
} // namespace elman
