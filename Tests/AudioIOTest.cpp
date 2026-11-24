#include "IO/Json.h"
#include "IO/Wave.h"
#include "Text/Unicode.h"
#include "Text/WhisperTokenizer.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    check(caught, "Invalid audio input was accepted");
}
void integer(std::ostream& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(char(value >> (8 * i)));
}
void wave(const std::filesystem::path& path, int format, int bits, const std::vector<uint32_t>& values, bool extensible = false) {
    const int bytes = bits / 8, data_size = int(values.size()) * bytes, fmt_size = extensible ? 40 : 16;
    std::ofstream out(path, std::ios::binary);
    out.write("RIFF", 4); integer(out, 4 + 8 + fmt_size + 8 + data_size + (data_size & 1), 4); out.write("WAVEfmt ", 8);
    integer(out, fmt_size, 4); integer(out, extensible ? 0xfffe : format, 2); integer(out, 2, 2);
    integer(out, 16000, 4); integer(out, 16000 * bytes * 2, 4); integer(out, bytes * 2, 2); integer(out, bits, 2);
    if (extensible) {
        integer(out, 22, 2); integer(out, bits, 2); integer(out, 3, 4);
        integer(out, format, 4); integer(out, 0, 2); integer(out, 0x10, 2);
        integer(out, 0x719b3800aa000080ULL, 8);
    }
    out.write("data", 4); integer(out, data_size, 4);
    for (auto value : values) integer(out, value, bytes);
    if (data_size & 1) out.put(0);
}
void expect(const std::filesystem::path& path, const std::vector<float>& expected) {
    const auto result = io::read_wave(path);
    check(result.sample_rate == 16000 && result.samples.size() == expected.size(), "Unexpected WAV dimensions");
    for (size_t i = 0; i < expected.size(); ++i)
        check(std::abs(result.samples[i] - expected[i]) < 1e-6, "Incorrect WAV sample decoding");
}
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "Usage: elman_audio_io_tests DIRECTORY");
        const auto directory = std::filesystem::u8path(argv[1]);
        std::filesystem::create_directories(directory);
        const auto path = directory / "audio.wav";
        wave(path, 1, 8, {0, 128, 255, 129}); expect(path, {-.5f, .5f});
        wave(path, 1, 16, {0x8000, 0, 0x4000, 0x4000}); expect(path, {-.5f, .5f});
        wave(path, 1, 24, {0x800000, 0, 0x400000, 0x400000}); expect(path, {-.5f, .5f});
        wave(path, 1, 32, {0x80000000, 0, 0x40000000, 0x40000000}, true); expect(path, {-.5f, .5f});
        wave(path, 3, 32, {0xbf800000, 0, 0x3f000000, 0x3f000000}, true); expect(path, {-.5f, .5f});
        wave(path, 3, 32, {0x7f800000, 0}); rejects([&] { io::read_wave(path); });
        wave(path, 1, 16, {0}); rejects([&] { io::read_wave(path); });
        wave(path, 6, 8, {0, 0}); rejects([&] { io::read_wave(path); });
        wave(path, 1, 16, {0, 0});
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
        rejects([&] { io::read_wave(path); });

        Json::Array vocabulary;
        for (char16_t codepoint : bytelevel_codepoints()) vocabulary.emplace_back(utf8(std::u16string(1, codepoint)));
        vocabulary.emplace_back("");
        vocabulary.emplace_back("<|endoftext|>"); vocabulary.emplace_back("<|0.00|>");
        Json(vocabulary).write((directory / "vocabulary.json").u8string());
        const WhisperTokenizer tokenizer(directory);
        check(tokenizer.end() == 257 && tokenizer.timestamp_begin() == 258, "Lost Whisper special IDs");
        check(tokenizer.decode({72, 105, 32, 0xe4, 0xb8, 0xad, 256, 257, 258}) == "Hi \xe4\xb8\xad", "Whisper ByteLevel decoding failed");
        check(tokenizer.bytes(0xe4) == std::string(1, char(0xe4)), "Partial UTF-8 token was changed");
        check(tokenizer.decode({0xe4}) == "\xef\xbf\xbd", "Invalid UTF-8 was not replaced");
        rejects([&] { tokenizer.decode({-1}); }); rejects([&] { tokenizer.token("missing"); });
        vocabulary.emplace_back("<|endoftext|>");
        Json(vocabulary).write((directory / "vocabulary.json").u8string());
        rejects([&] { WhisperTokenizer invalid(directory); });
        std::cout << "Audio IO and Whisper tokenizer checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
