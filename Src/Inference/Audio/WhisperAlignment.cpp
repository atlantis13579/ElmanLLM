#include "Inference/Audio/WhisperAlignment.h"

#include "Text/Unicode.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
int reflect(int index, int size) {
    if (size == 1) return 0;
    const int period = 2 * size - 2;
    index = (index % period + period) % period;
    return index < size ? index : period - index;
}

// Align each prediction row to monotonically increasing audio frames.
std::vector<int> boundaries(const std::vector<float>& matrix, int rows, int frames) {
    const int stride = frames + 1;
    std::vector<double> cost(size_t(rows + 1) * stride, std::numeric_limits<double>::infinity());
    std::vector<uint8_t> direction(cost.size(), 0);
    cost[0] = 0;
    for (int i = 1; i <= rows; ++i) for (int j = 1; j <= frames; ++j) {
        const size_t position = size_t(i) * stride + j;
        const double diagonal = cost[position - stride - 1], vertical = cost[position - stride], horizontal = cost[position - 1];
        uint8_t step = diagonal < vertical && diagonal < horizontal ? 0 : vertical < diagonal && vertical < horizontal ? 1 : 2;
        cost[position] = (step == 0 ? diagonal : step == 1 ? vertical : horizontal) - matrix[size_t(i - 1) * frames + j - 1];
        direction[position] = step;
    }
    std::vector<int> result(rows, frames - 1);
    int row = rows, frame = frames;
    while (row > 0 && frame > 0) {
        result[row - 1] = frame - 1;
        const auto step = direction[size_t(row) * stride + frame];
        if (step == 0 || step == 1) --row;
        if (step == 0 || step == 2) --frame;
    }
    return result;
}
}

std::vector<WhisperAlignedWord> align_whisper_words(const Whisper& model, const WhisperMemory& memory,
    const WhisperTokenizer& tokenizer, const std::vector<int32_t>& prompt,
    const std::vector<int32_t>& text, const std::vector<std::pair<int, int>>& heads,
    int audio_frames, const std::string& language, const std::vector<size_t>& segment_ends) {
    if (text.empty()) return {};
    if (heads.empty() || audio_frames <= 0 || audio_frames > model.config().audio_context)
        throw std::invalid_argument("Invalid Whisper alignment dimensions");
    std::vector<int32_t> tokens = prompt;
    tokens.push_back(tokenizer.token("<|notimestamps|>"));
    tokens.insert(tokens.end(), text.begin(), text.end()); tokens.push_back(tokenizer.end());
    const auto output = model.decode(tokens, memory, nullptr, heads);
    auto scores = slice(output.alignment, 2, 0, audio_frames).data();
    const int token_count = int(tokens.size()), rows = int(text.size()) + 1;
    const size_t head_stride = size_t(token_count) * audio_frames;
    std::vector<float> matrix(size_t(rows) * audio_frames, 0);
    for (size_t head = 0; head < heads.size(); ++head) {
        float* values = scores.data() + head * head_stride;
        for (int token = 0; token < token_count; ++token) {
            float* row = values + size_t(token) * audio_frames;
            const float peak = *std::max_element(row, row + audio_frames);
            double total = 0;
            for (int frame = 0; frame < audio_frames; ++frame) { row[frame] = std::exp(row[frame] - peak); total += row[frame]; }
            for (int frame = 0; frame < audio_frames; ++frame) row[frame] = float(row[frame] / total);
        }
        for (int frame = 0; frame < audio_frames; ++frame) {
            double sum = 0, variance = 0;
            for (int token = 0; token < token_count; ++token) sum += values[size_t(token) * audio_frames + frame];
            const double mean = sum / token_count;
            for (int token = 0; token < token_count; ++token) {
                const double difference = values[size_t(token) * audio_frames + frame] - mean;
                variance += difference * difference;
            }
            const double scale = 1 / std::max(1e-8, std::sqrt(variance / token_count));
            for (int token = 0; token < token_count; ++token)
                values[size_t(token) * audio_frames + frame] = float((values[size_t(token) * audio_frames + frame] - mean) * scale);
        }
        for (int row = 0; row < rows; ++row) for (int frame = 0; frame < audio_frames; ++frame) {
            std::array<float, 7> window;
            for (int k = 0; k < 7; ++k)
                window[k] = values[(prompt.size() + row) * audio_frames + reflect(frame + k - 3, audio_frames)];
            std::nth_element(window.begin(), window.begin() + 3, window.end());
            matrix[size_t(row) * audio_frames + frame] += window[3] / float(heads.size());
        }
    }
    const auto times = boundaries(matrix, rows, audio_frames);
    const bool unspaced = language == "zh" || language == "ja" || language == "th" || language == "lo" || language == "my" || language == "yue";
    std::vector<WhisperAlignedWord> words;
    std::string piece;
    int first = 0;
    for (int token = 0; token < int(text.size()); ++token) {
        piece += tokenizer.bytes(text[token]);
        if (replace_invalid_utf8(piece) != piece && token + 1 < int(text.size())) continue;
        const bool space = !piece.empty() && (piece.front() == ' ' || piece.front() == '\n' || piece.front() == '\t');
        const bool boundary = std::binary_search(segment_ends.begin(), segment_ends.end(), size_t(first));
        if (!piece.empty()) {
            if (words.empty() || space || unspaced || boundary)
                words.push_back({{times[first] * .02, times[token + 1] * .02, replace_invalid_utf8(piece)}, size_t(token + 1)});
            else {
                words.back().word.text += replace_invalid_utf8(piece);
                words.back().word.end = times[token + 1] * .02;
                words.back().token_end = size_t(token + 1);
            }
        }
        piece.clear(); first = token + 1;
    }
    std::vector<double> durations;
    for (const auto& item : words) if (item.word.end > item.word.start) durations.push_back(item.word.end - item.word.start);
    if (!durations.empty()) {
        std::sort(durations.begin(), durations.end());
        const double maximum = 2 * std::min(.7, durations[durations.size() / 2]);
        for (auto& item : words) if (item.word.end - item.word.start > maximum)
            item.word.start = std::max(item.word.start, item.word.end - maximum);
    }
    return words;
}

} // namespace elman
