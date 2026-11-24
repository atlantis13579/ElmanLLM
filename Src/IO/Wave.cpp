#include "IO/Wave.h"

#include "IO/BinaryInput.h"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace elman::io {

WaveAudio read_wave(const std::filesystem::path& path) {
    BinaryInput file(path);
    require(file.text(4) == "RIFF", "Audio input must be a RIFF WAV file");
    const uint64_t riff_bytes = file.le(4);
    require(riff_bytes >= 4 && riff_bytes <= file.remaining(), "Truncated RIFF audio container");
    const uint64_t end = file.position() + riff_bytes;
    require(file.text(4) == "WAVE", "Audio input must be a WAV file");
    int format = 0, channels = 0, rate = 0, width = 0, alignment = 0;
    uint64_t data_offset = 0, data_bytes = 0;
    bool has_format = false, has_data = false;
    while (file.position() < end) {
        require(end - file.position() >= 8, "Truncated WAV chunk header");
        const auto key = file.text(4);
        const uint64_t bytes = file.le(4), begin = file.position();
        require(bytes <= end - begin && (bytes + (bytes & 1)) <= end - begin, "Truncated WAV chunk");
        if (key == "fmt ") {
            require(!has_format && bytes >= 16, "Invalid or duplicate WAV format");
            has_format = true;
            format = int(file.le(2)); channels = int(file.le(2));
            const uint64_t sample_rate = file.le(4), byte_rate = file.le(4);
            alignment = int(file.le(2)); width = int(file.le(2));
            require(sample_rate > 0 && sample_rate <= 384000 && channels > 0 && channels <= 64,
                    "Invalid WAV channel count or sample rate");
            rate = int(sample_rate);
            if (format == 0xfffe) {
                require(bytes >= 40 && file.le(2) >= 22, "Invalid extensible WAV format");
                const uint64_t valid_bits = file.le(2);
                file.skip(4);
                const auto guid = file.bytes(16);
                const uint8_t tail[14] = {0,0,0,0,0x10,0,0x80,0,0,0xaa,0,0x38,0x9b,0x71};
                require(std::memcmp(guid.data() + 2, tail, sizeof(tail)) == 0 && valid_bits == uint64_t(width),
                        "Unsupported extensible WAV sample format");
                format = int(guid[0]) | (int(guid[1]) << 8);
            }
            require((format == 1 && (width == 8 || width == 16 || width == 24 || width == 32)) ||
                    (format == 3 && width == 32), "WAV input must contain PCM or IEEE Float32 samples");
            require(alignment == channels * (width / 8) && byte_rate == uint64_t(rate) * alignment,
                    "Invalid WAV block alignment or byte rate");
        } else if (key == "data") {
            require(!has_data, "Multiple WAV data chunks are unsupported");
            has_data = true; data_offset = begin; data_bytes = bytes;
        }
        file.seek(begin + bytes + (bytes & 1));
    }
    require(has_format && has_data && data_bytes % alignment == 0, "Missing or incomplete WAV samples");
    require(data_bytes / alignment <= std::numeric_limits<size_t>::max(), "WAV input is too large");
    WaveAudio audio;
    audio.sample_rate = rate;
    audio.samples.resize(size_t(data_bytes / alignment));
    file.seek(data_offset);
    const int sample_bytes = width / 8;
    std::vector<uint8_t> block(size_t(alignment) * 4096);
    for (size_t begin = 0; begin < audio.samples.size(); begin += 4096) {
        const size_t frames = std::min<size_t>(4096, audio.samples.size() - begin);
        file.read(block.data(), frames * alignment);
        for (size_t i = 0; i < frames; ++i) {
            double value = 0;
            for (int channel = 0; channel < channels; ++channel) {
                const uint32_t bits = uint32_t(view(block).le(i * alignment + channel * sample_bytes, sample_bytes));
                if (format == 3) {
                    float sample;
                    std::memcpy(&sample, &bits, sizeof(sample));
                    require(std::isfinite(sample), "WAV contains a non-finite floating sample");
                    value += sample;
                } else if (width == 8) value += (int(bits) - 128) / 128.;
                else {
                    const int64_t sign = int64_t(1) << (width - 1);
                    const int64_t sample = (int64_t(bits) ^ sign) - sign;
                    value += double(sample) / sign;
                }
            }
            audio.samples[begin + i] = float(value / channels);
        }
    }
    return audio;
}

} // namespace elman::io
