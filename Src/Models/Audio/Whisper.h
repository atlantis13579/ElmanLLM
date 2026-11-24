#pragma once

#include "Core/Tensor.h"
#include <filesystem>
#include <memory>

namespace elman {

struct WhisperConfig {
    int mel_bins = 0, audio_context = 0, audio_width = 0, audio_heads = 0, audio_layers = 0;
    int vocabulary = 0, text_context = 0, text_width = 0, text_heads = 0, text_layers = 0;
};

struct WhisperAttentionCache { Tensor key, value; };
struct WhisperMemory { std::vector<WhisperAttentionCache> layers; };
struct WhisperCache { std::vector<WhisperAttentionCache> layers; int length = 0; };
struct WhisperOutput { Tensor logits; WhisperCache cache; Tensor alignment; };

// OpenAI Whisper encoder/decoder with native CTranslate2 weight import.
// Model weights and encoded memory are immutable. Decoder caches belong to the caller.
class Whisper {
public:
    Whisper(const std::filesystem::path& directory, Backend backend, DType dtype = DType::Float32);
    ~Whisper();
    Whisper(const Whisper&) = delete;
    Whisper& operator=(const Whisper&) = delete;
    const WhisperConfig& config() const;
    Backend backend() const;
    DType dtype() const;
    int device() const;
    Tensor encode(const Tensor& mel, const std::function<bool(int, int)>& progress = {}) const;
    WhisperMemory prepare_memory(const Tensor& encoded) const;
    WhisperOutput decode(const std::vector<int32_t>& tokens, const WhisperMemory& memory,
                          const WhisperCache* past = nullptr,
                          const std::vector<std::pair<int, int>>& alignment_heads = {}) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elman
