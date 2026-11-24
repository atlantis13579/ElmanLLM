#include "Inference/Audio/WhisperSession.h"
#include "Inference/Audio/WhisperAlignment.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "Core/AllocationPool.h"
#include "Core/Device.h"
#include "IO/Json.h"
#include "Models/Audio/Whisper.h"
#include "Operators/audio_ops.h"
#include "Text/WhisperTokenizer.h"

namespace elman {
namespace {
constexpr float excluded = -std::numeric_limits<float>::infinity();
void report(const SpeechProgress& progress, double fraction, const std::string& message) {
    if (progress && !progress(std::clamp(fraction, 0., 1.), message))
        throw std::runtime_error("Speech recognition cancelled");
}
std::string trim(std::string value) {
    const auto first = value.find_first_not_of(" \t\r\n"), last = value.find_last_not_of(" \t\r\n");
    return first == std::string::npos ? std::string{} : value.substr(first, last - first + 1);
}
double log_sum(const std::vector<float>& logits, size_t first, size_t last) {
    if (first == last) return excluded;
    const float maximum = *std::max_element(logits.begin() + first, logits.begin() + last);
    if (!std::isfinite(maximum)) return excluded;
    double total = 0;
    for (size_t i = first; i < last; ++i) total += std::exp(double(logits[i]) - maximum);
    return maximum + std::log(total);
}
struct Hypothesis {
    std::vector<int32_t> tokens;
    double score = 0;
    WhisperCache cache;
};
struct DecodedWindow {
    std::vector<int32_t> tokens, prompt;
    std::string language;
    double no_speech = 0, average_logprob = 0;
};
}

struct WhisperSession::Impl {
    Whisper model;
    WhisperTokenizer tokenizer;
    LogMelSpectrogram spectrogram;
    std::vector<int32_t> suppressed, suppressed_begin, language_ids;
    std::vector<std::pair<int, int>> alignment_heads;
    int start, task, no_timestamps, no_speech;

    Impl(const std::filesystem::path& directory, Backend backend, DType dtype)
        : model(directory, backend, dtype), tokenizer(directory), spectrogram(backend, model.config().mel_bins),
          start(tokenizer.token("<|startoftranscript|>")), task(tokenizer.token("<|transcribe|>")),
          no_timestamps(tokenizer.token("<|notimestamps|>")), no_speech(tokenizer.token("<|nospeech|>")) {
        if (tokenizer.size() != size_t(model.config().vocabulary))
            throw std::invalid_argument("Whisper model and vocabulary sizes differ");
        const Json config = Json::read(directory / "config.json");
        auto ids = [&](const char* key) {
            std::vector<int32_t> result;
            if (config.contains(key)) for (const Json& item : config.at(key).array()) {
                const int id = item.integer();
                if (id < 0 || id >= model.config().vocabulary) throw std::invalid_argument("Invalid Whisper configuration token ID");
                result.push_back(id);
            }
            return result;
        };
        suppressed = ids("suppress_ids"); suppressed_begin = ids("suppress_ids_begin"); language_ids = ids("lang_ids");
        for (int id : {start, task, no_speech, no_timestamps}) suppressed.push_back(id);
        if (config.contains("alignment_heads")) for (const Json& pair : config.at("alignment_heads").array()) {
            if (pair.array().size() != 2) throw std::invalid_argument("Invalid Whisper alignment head pair");
            const int layer = pair.array()[0].integer(), head = pair.array()[1].integer();
            if (layer < 0 || layer >= model.config().text_layers || head < 0 || head >= model.config().text_heads)
                throw std::invalid_argument("Invalid Whisper alignment head index");
            alignment_heads.emplace_back(layer, head);
        }
        if (alignment_heads.empty()) for (int layer = model.config().text_layers / 2; layer < model.config().text_layers; ++layer)
            for (int head = 0; head < model.config().text_heads; ++head) alignment_heads.emplace_back(layer, head);
    }

    void filter(std::vector<float>& logits, const std::vector<int32_t>& tokens) const {
        for (float value : logits) if (std::isnan(value) || value == std::numeric_limits<float>::infinity())
            throw std::runtime_error("Whisper produced non-finite logits");
        for (int id : suppressed) logits[id] = excluded;
        const int timestamp = tokenizer.timestamp_begin(), end = tokenizer.end();
        if (tokens.empty()) {
            for (int id : suppressed_begin) logits[id] = excluded;
            std::fill(logits.begin(), logits.begin() + timestamp, excluded);
            std::fill(logits.begin() + std::min<size_t>(timestamp + 51, logits.size()), logits.end(), excluded);
        } else {
            const bool last_time = tokens.back() >= timestamp;
            const bool previous_time = tokens.size() < 2 || tokens[tokens.size() - 2] >= timestamp;
            if (last_time) {
                if (previous_time) std::fill(logits.begin() + timestamp, logits.end(), excluded);
                else std::fill(logits.begin(), logits.begin() + end, excluded);
            }
            auto found = std::find_if(tokens.rbegin(), tokens.rend(), [&](int id) { return id >= timestamp; });
            if (found != tokens.rend()) {
                const int minimum = *found + (last_time && !previous_time ? 0 : 1);
                std::fill(logits.begin() + timestamp, logits.begin() + std::min<size_t>(minimum, logits.size()), excluded);
            }
        }
        const double time_mass = log_sum(logits, timestamp, logits.size());
        const float text_maximum = *std::max_element(logits.begin(), logits.begin() + timestamp);
        if (time_mass > text_maximum) std::fill(logits.begin(), logits.begin() + timestamp, excluded);
    }

    DecodedWindow decode(const WhisperMemory& memory, const WhisperOptions& options,
                          const SpeechProgress& progress, double fraction) const {
        DecodedWindow result;
        auto initial = model.decode({start}, memory);
        const auto initial_logits = initial.logits.data();
        result.no_speech = std::exp(initial_logits[no_speech] - log_sum(initial_logits, 0, initial_logits.size()));
        result.language = options.language;
        if (options.language == "auto" && !language_ids.empty()) {
            const int id = *std::max_element(language_ids.begin(), language_ids.end(),
                [&](int a, int b) { return initial_logits[a] < initial_logits[b]; });
            const auto& language = tokenizer.bytes(id);
            result.language = language.substr(2, language.size() - 4);
        } else if (options.language == "auto") result.language = "en";
        std::vector<int32_t> prompt{start};
        if (!language_ids.empty()) {
            const int language = tokenizer.token("<|" + result.language + "|>");
            if (std::find(language_ids.begin(), language_ids.end(), language) == language_ids.end())
                throw std::invalid_argument("Unsupported Whisper source language");
            prompt.push_back(language); prompt.push_back(task);
        }
        auto first = model.decode(prompt, memory);
        std::vector<Hypothesis> active(1), finished;
        result.prompt = prompt;
        const int limit = std::min(options.max_tokens, model.config().text_context - int(prompt.size()) - 2);
        if (limit <= 0) throw std::invalid_argument("Whisper text context is too short");
        for (int step = 0; step < limit && !active.empty(); ++step) {
            report(progress, fraction, "Recognizing speech");
            std::vector<Hypothesis> candidates;
            for (const auto& beam : active) {
                auto output = step ? model.decode({beam.tokens.back()}, memory, &beam.cache) : first;
                auto logits = output.logits.data();
                filter(logits, beam.tokens);
                const double denominator = log_sum(logits, 0, logits.size());
                if (!std::isfinite(denominator)) throw std::runtime_error("Whisper has no valid next token");
                std::vector<int32_t> ordered(logits.size());
                std::iota(ordered.begin(), ordered.end(), 0);
                const size_t choices = std::min<size_t>(options.beam_size + 1, ordered.size());
                std::partial_sort(ordered.begin(), ordered.begin() + choices, ordered.end(), [&](int a, int b) {
                    return logits[a] == logits[b] ? a < b : logits[a] > logits[b];
                });
                for (size_t i = 0; i < choices && std::isfinite(logits[ordered[i]]); ++i) {
                    Hypothesis next{beam.tokens, beam.score + logits[ordered[i]] - denominator, output.cache};
                    next.tokens.push_back(ordered[i]); candidates.push_back(std::move(next));
                }
            }
            std::stable_sort(candidates.begin(), candidates.end(), [](const auto& a, const auto& b) { return a.score > b.score; });
            active.clear();
            for (auto& candidate : candidates) {
                if (candidate.tokens.back() == tokenizer.end()) finished.push_back(std::move(candidate));
                else if (active.size() < size_t(options.beam_size)) active.push_back(std::move(candidate));
                if (active.size() == size_t(options.beam_size)) break;
            }
            if (finished.size() >= size_t(options.beam_size)) break;
        }
        if (finished.empty()) finished = std::move(active);
        if (finished.empty()) throw std::runtime_error("Whisper decoding produced no hypotheses");
        const auto best = std::max_element(finished.begin(), finished.end(), [](const auto& a, const auto& b) {
            return a.score / a.tokens.size() < b.score / b.tokens.size();
        });
        result.tokens = best->tokens;
        result.average_logprob = best->score / best->tokens.size();
        return result;
    }
};

WhisperSession::WhisperSession(const std::filesystem::path& directory, Backend backend, DType dtype)
    : impl_(std::make_unique<Impl>(directory, backend, dtype)) {}
WhisperSession::~WhisperSession() = default;

Transcription WhisperSession::transcribe(const std::vector<float>& samples, const WhisperOptions& options,
                                         const SpeechProgress& progress) {
    if (options.beam_size < 1 || options.beam_size > 16 || options.max_tokens < 1 || options.max_tokens > 448)
        throw std::invalid_argument("Invalid Whisper beam width or token limit");
    for (float value : samples) if (!std::isfinite(value)) throw std::invalid_argument("Non-finite audio sample");
    DeviceScope owner(impl_->model.backend(), impl_->model.device());
    AllocationPool pool(impl_->model.backend(), 512 * 1024 * 1024);
    Transcription result;
    result.language = options.language == "auto" ? "en" : options.language;
    const size_t window_size = size_t(impl_->model.config().audio_context) * 320;
    const int timestamp = impl_->tokenizer.timestamp_begin();
    for (size_t seek = 0; seek < samples.size();) {
        const size_t available = std::min(window_size, samples.size() - seek);
        const double fraction = double(seek) / samples.size(), offset = seek / 16000., duration = available / 16000.;
        report(progress, fraction, "Encoding audio");
        std::vector<float> window(window_size, 0);
        std::copy_n(samples.begin() + seek, available, window.begin());
        double energy = 0;
        for (float value : window) energy += double(value) * value;
        if (energy / available < 1e-12) { seek += available; continue; }
        const auto mel = impl_->spectrogram(Tensor::from_data({int(window_size)}, window, impl_->model.backend()));
        const auto encoded = impl_->model.encode(mel, [&](int layer, int count) {
            report(progress, fraction, "Encoding audio " + std::to_string(layer + 1) + "/" + std::to_string(count)); return true;
        });
        const auto memory = impl_->model.prepare_memory(encoded);
        const auto decoded = impl_->decode(memory, options, progress, fraction);
        result.language = decoded.language;
        if (decoded.no_speech > .6 && decoded.average_logprob < -1) { seek += available; continue; }
        std::vector<int32_t> text, all_text;
        std::vector<size_t> segment_ends;
        const size_t first_segment = result.segments.size();
        double begin = 0, last_end = 0;
        bool closed = false;
        auto append = [&](double end) {
            const auto value = trim(impl_->tokenizer.decode(text));
            end = std::clamp(end, begin, duration);
            if (!value.empty() && end > begin) {
                result.segments.push_back({offset + begin, offset + end, value, {}});
                all_text.insert(all_text.end(), text.begin(), text.end());
                segment_ends.push_back(all_text.size());
            }
            text.clear(); last_end = end;
        };
        for (int id : decoded.tokens) {
            if (id >= timestamp) {
                const double time = std::clamp((id - timestamp) * .02, 0., duration);
                if (!text.empty()) { append(time); closed = true; }
                else { begin = time; closed = false; }
            } else if (id < impl_->tokenizer.end()) { text.push_back(id); closed = false; }
        }
        size_t advance = available;
        if (!closed && last_end > 0)
            advance = std::min(available, std::max<size_t>(160, size_t(last_end * 16000)));
        else if (!text.empty()) append(duration);
        if (options.word_timestamps && !all_text.empty()) {
            report(progress, fraction, "Aligning words");
            const auto words = align_whisper_words(impl_->model, memory, impl_->tokenizer, decoded.prompt,
                all_text, impl_->alignment_heads, std::max(1, int((available + 319) / 320)), decoded.language, segment_ends);
            size_t segment = 0;
            for (const auto& item : words) {
                while (segment + 1 < segment_ends.size() && item.token_end > segment_ends[segment]) ++segment;
                SpeechWord word = item.word;
                word.start = offset + std::clamp(word.start, 0., duration);
                word.end = offset + std::clamp(word.end, word.start - offset, duration);
                result.segments[first_segment + segment].words.push_back(std::move(word));
            }
            for (size_t i = first_segment; i < result.segments.size(); ++i) {
                auto& item = result.segments[i];
                if (!item.words.empty()) {
                    item.start = item.words.front().start;
                    item.end = item.words.back().end;
                }
            }
        }
        seek += advance;
    }
    report(progress, 1, "Speech recognition complete");
    return result;
}

} // namespace elman
