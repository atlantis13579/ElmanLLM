#include "Commands/Commands.h"

#include "IO/AtomicFile.h"
#include "IO/Json.h"
#include "IO/Wave.h"
#include "Inference/Audio/WhisperSession.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace elman {
namespace {
int positive(const std::string& value) {
    size_t consumed = 0;
    const int result = std::stoi(value, &consumed);
    if (consumed != value.size() || result <= 0) throw std::invalid_argument("Expected a positive integer");
    return result;
}
std::string timestamp(double seconds) {
    const int64_t ms = int64_t(seconds * 1000 + .5);
    std::ostringstream text;
    text << std::setfill('0') << std::setw(2) << ms / 3600000 << ':'
         << std::setw(2) << ms / 60000 % 60 << ':' << std::setw(2) << ms / 1000 % 60 << ',' << std::setw(3) << ms % 1000;
    return text.str();
}
}

int transcribe_command(int argc, char** argv) {
    if (argc < 3) throw std::invalid_argument("transcribe requires a Whisper model directory");
    std::filesystem::path input, output;
    Backend backend = Backend::CPU;
    std::string precision;
    WhisperOptions options;
    for (int i = 3; i < argc; ++i) {
        const std::string key = argv[i];
        if (++i == argc) throw std::invalid_argument("Missing value for " + key);
        const std::string value = argv[i];
        if (key == "--input") input = std::filesystem::u8path(value);
        else if (key == "--output") output = std::filesystem::u8path(value);
        else if (key == "--backend") backend = parse_backend(value);
        else if (key == "--dtype") precision = value;
        else if (key == "--language") options.language = value;
        else if (key == "--beam-size") options.beam_size = positive(value);
        else if (key == "--max-tokens") options.max_tokens = positive(value);
        else throw std::invalid_argument("Unknown transcribe option " + key);
    }
    if (input.empty() || output.empty()) throw std::invalid_argument("transcribe requires --input and --output");
    const auto extension = output.extension().u8string();
    if (extension != ".json" && extension != ".srt") throw std::invalid_argument("Transcription output must be JSON or SRT");
    const auto audio = io::read_wave(input);
    if (audio.sample_rate != 16000) throw std::invalid_argument("Whisper requires 16 kHz WAV; convert audio with ffmpeg -ar 16000 -ac 1");
    const DType dtype = precision.empty() ? (backend == Backend::CPU ? DType::Float32 : DType::Float16) : parse_dtype(precision);
    WhisperSession session(std::filesystem::u8path(argv[2]), backend, dtype);
    std::string previous;
    const auto result = session.transcribe(audio.samples, options, [&](double fraction, const std::string& message) {
        if (message != previous) std::cerr << int(fraction * 100) << "% " << message << '\n';
        previous = message; return true;
    });
    if (!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    write_file_atomically(output, [&](const std::filesystem::path& temporary) {
        if (extension == ".json") {
            Json::Array segments;
            for (const auto& item : result.segments) {
                Json::Array words;
                for (const auto& word : item.words)
                    words.emplace_back(Json::Object{{"start", word.start}, {"end", word.end}, {"text", word.text}});
                segments.emplace_back(Json::Object{{"start", item.start}, {"end", item.end}, {"text", item.text}, {"words", std::move(words)}});
            }
            Json(Json::Object{{"language", result.language}, {"segments", std::move(segments)}}).write(temporary.u8string());
        } else {
            std::ofstream file(temporary, std::ios::binary);
            for (size_t i = 0; i < result.segments.size(); ++i) {
                const auto& segment = result.segments[i];
                file << i + 1 << '\n' << timestamp(segment.start) << " --> " << timestamp(segment.end) << '\n'
                     << segment.text << "\n\n";
            }
            if (!file) throw std::runtime_error("Cannot write transcription output");
        }
    });
    std::cout << output.u8string() << '\n';
    return 0;
}

} // namespace elman
