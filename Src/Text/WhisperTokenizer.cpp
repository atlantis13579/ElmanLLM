#include "Text/WhisperTokenizer.h"

#include "IO/Json.h"
#include "Text/Unicode.h"
#include <limits>
#include <stdexcept>

namespace elman {

WhisperTokenizer::WhisperTokenizer(const std::filesystem::path& directory) {
    const Json vocabulary = Json::read(directory / "vocabulary.json");
    if (vocabulary.array().size() > size_t(std::numeric_limits<int32_t>::max()))
        throw std::invalid_argument("Whisper vocabulary exceeds the token ID range");
    for (const Json& entry : vocabulary.array()) {
        const std::string text = entry.string();
        if (!tokens_.emplace(text, int32_t(bytes_.size())).second)
            throw std::invalid_argument("Whisper vocabulary contains a duplicate token");
        bytes_.push_back(text);
    }
    end_ = token("<|endoftext|>");
    timestamp_ = token("<|0.00|>");
    std::map<char16_t, uint8_t> alphabet;
    const auto& codepoints = bytelevel_codepoints();
    for (int i = 0; i < 256; ++i) alphabet.emplace(codepoints[i], uint8_t(i));
    for (int32_t i = 0; i < end_; ++i) {
        std::string decoded;
        for (char16_t codepoint : utf16(bytes_[i])) {
            const auto found = alphabet.find(codepoint);
            if (found == alphabet.end()) throw std::invalid_argument("Whisper vocabulary contains invalid ByteLevel text");
            decoded.push_back(char(found->second));
        }
        bytes_[i] = std::move(decoded);
    }
}

int32_t WhisperTokenizer::token(const std::string& spelling) const {
    const auto found = tokens_.find(spelling);
    if (found == tokens_.end()) throw std::invalid_argument("Unknown Whisper token: " + spelling);
    return found->second;
}

const std::string& WhisperTokenizer::bytes(int32_t id) const {
    if (id < 0 || size_t(id) >= bytes_.size()) throw std::out_of_range("Whisper token ID is outside the vocabulary");
    return bytes_[id];
}

std::string WhisperTokenizer::decode(const std::vector<int32_t>& ids) const {
    std::string result;
    for (int32_t id : ids) {
        const auto& value = bytes(id);
        if (id < end_) result += value;
    }
    return replace_invalid_utf8(result);
}

} // namespace elman
