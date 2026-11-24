#pragma once

#include <filesystem>
#include <vector>

namespace elman::io {

struct WaveAudio {
    int sample_rate = 0;
    std::vector<float> samples;
};

// Reads uncompressed PCM or IEEE Float32 WAV and downmixes channels to mono.
WaveAudio read_wave(const std::filesystem::path& path);

} // namespace elman::io
