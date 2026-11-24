#pragma once

#include "Core/Tensor.h"
#include <filesystem>
#include <memory>

namespace elman {

struct SpeechWord { double start = 0, end = 0; std::string text; };
struct SpeechSegment {
    double start = 0, end = 0;
    std::string text;
    std::vector<SpeechWord> words;
};
struct Transcription { std::string language; std::vector<SpeechSegment> segments; };
struct WhisperOptions {
    std::string language = "auto";
    int beam_size = 5, max_tokens = 224;
    bool word_timestamps = true;
};
using SpeechProgress = std::function<bool(double, const std::string&)>;

// Processes mono 16 kHz samples in bounded windows, with timestamp-constrained beam search.
// Return false from progress to cancel. Sessions are owned by one inference thread.
class WhisperSession {
public:
    WhisperSession(const std::filesystem::path& directory, Backend backend, DType dtype = DType::Float32);
    ~WhisperSession();
    Transcription transcribe(const std::vector<float>& samples, const WhisperOptions& options = {},
                             const SpeechProgress& progress = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace elman
