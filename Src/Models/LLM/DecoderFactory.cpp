#include "Models/LLM/DecoderFactory.h"
#include "Models/LLM/Qwen2.h"
#include "Models/LLM/DeepSeek.h"
#include "Models/LLM/Qwen35.h"
#include "Core/Device.h"
#include <filesystem>
#include <stdexcept>

namespace elman {
DecoderConfig read_decoder_config(const std::string& directory) {
    const auto gguf = find_gguf_model(std::filesystem::u8path(directory));
    if (!gguf.empty()) return Qwen35Config::from_gguf(GgufFile(gguf));
    Json value = Json::read((std::filesystem::path(directory) / "config.json").string());
    const std::string type = value.at("model_type").string();
    if (type == "qwen2") return Qwen2Config::from_json(value);
    if (type == "deepseek_v2" || type == "deepseek_v3") return DeepSeekConfig::from_json(value);
    throw std::invalid_argument("Unsupported decoder model_type: " + type);
}
std::unique_ptr<DecoderModel> load_decoder(const std::string& directory, Backend backend, DType dtype,
                                         const DecoderLoadOptions& options) {
    const auto gguf = find_gguf_model(std::filesystem::u8path(directory));
    if (!gguf.empty()) {
        if (options.tensor_parallel || options.devices.size() > 1 || options.simulated_devices > 1 || options.direct_paged_attention)
            throw std::invalid_argument("Qwen3.5 uses a single-device hybrid recurrent cache");
        DeviceScope owner(backend, options.devices.empty() ? current_device(backend) : options.devices[0]);
        return std::make_unique<Qwen35>(gguf, backend, dtype);
    }
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
