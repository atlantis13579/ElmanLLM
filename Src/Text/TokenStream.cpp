#include "Text/TokenStream.h"

#include <algorithm>
#include <stdexcept>

namespace elman {
namespace {
size_t complete_prefix(const std::string& text) {
    size_t position = 0;
    while (position < text.size()) {
        unsigned char lead = static_cast<unsigned char>(text[position]);
        int width = lead >= 0xc2 && lead <= 0xdf ? 2 : lead >= 0xe0 && lead <= 0xef ? 3 :
                    lead >= 0xf0 && lead <= 0xf4 ? 4 : 1;
        if (width == 1) { ++position; continue; }
        int next = 1;
        for (; next < width; ++next) {
            if (position + next == text.size()) return position;
            unsigned char byte = static_cast<unsigned char>(text[position + next]);
            if (byte < 0x80 || byte > 0xbf || (next == 1 &&
                ((lead == 0xe0 && byte < 0xa0) ||
                 (lead == 0xf0 && byte < 0x90) || (lead == 0xf4 && byte > 0x8f)))) break;
        }
        // Hold incomplete sequences. As in CPython's incremental UTF-8 codec,
        // defer surrogate rejection (ED A0..BF) until the third byte or finish.
        // ICU still replaces each invalid byte once that sequence is complete.
        position += next;
    }
    return position;
}
}

std::string TokenizerStream::push(int32_t id) {
    if (finished_) throw std::logic_error("Cannot push into a finished tokenizer stream");
    if (id < 0) throw std::invalid_argument("Negative token ID");
    if (size_t(id) >= tokenizer_.decoded_.size() || (skip_special_ && tokenizer_.special_[id])) return {};
    pending_ += tokenizer_.decoded_[id];
    size_t length = complete_prefix(pending_);
    std::string result = replace_invalid_utf8(pending_.substr(0, length));
    pending_.erase(0, length);
    return result;
}

std::string TokenizerStream::finish() {
    if (finished_) return {};
    finished_ = true;
    std::string result = replace_invalid_utf8(pending_);
    pending_.clear();
    return result;
}

TextStopFilter::TextStopFilter(std::vector<std::string> stops) : stops_(std::move(stops)) {
    for (const auto& stop : stops_) {
        if (stop.empty()) throw std::invalid_argument("Stop strings must be nonempty");
        utf16(stop);
    }
}

std::string TextStopFilter::push(const std::string& text) {
    if (finished_) throw std::logic_error("Cannot push into a finished stop filter");
    if (stopped_) return {};
    utf16(text);
    pending_ += text;
    size_t first = std::string::npos;
    for (const auto& stop : stops_) {
        size_t index = pending_.find(stop);
        if (index < first) { first = index; matched_ = stop; }
    }
    if (first != std::string::npos) {
        std::string result = pending_.substr(0, first);
        pending_.clear();
        stopped_ = true;
        return result;
    }
    size_t hold = 0;
    for (const auto& stop : stops_) {
        for (size_t length = std::min(pending_.size(), stop.size() - 1); length > hold; --length) {
            if (pending_.compare(pending_.size() - length, length, stop, 0, length) == 0) {
                hold = length;
                break;
            }
        }
    }
    std::string result = pending_.substr(0, pending_.size() - hold);
    pending_.erase(0, result.size());
    return result;
}

std::string TextStopFilter::finish() {
    if (finished_) return {};
    finished_ = true;
    std::string result = std::move(pending_);
    pending_.clear();
    return result;
}

} // namespace elman
