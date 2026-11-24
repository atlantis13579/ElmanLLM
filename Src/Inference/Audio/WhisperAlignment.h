#pragma once

#include "Inference/Audio/WhisperSession.h"
#include "Models/Audio/Whisper.h"
#include "Text/WhisperTokenizer.h"

namespace elman {

struct WhisperAlignedWord { SpeechWord word; size_t token_end = 0; };

std::vector<WhisperAlignedWord> align_whisper_words(const Whisper& model, const WhisperMemory& memory,
    const WhisperTokenizer& tokenizer, const std::vector<int32_t>& prompt,
    const std::vector<int32_t>& text, const std::vector<std::pair<int, int>>& heads,
    int audio_frames, const std::string& language, const std::vector<size_t>& segment_ends = {});

} // namespace elman
