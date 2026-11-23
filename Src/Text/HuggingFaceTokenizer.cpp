#include "Text/HuggingFaceTokenizer.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>

namespace elman {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("HF tokenizer: " + message);
}
uint64_t pair_key(int32_t left, int32_t right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}
std::map<char16_t, uint8_t> byte_decoder() {
    std::map<char16_t, uint8_t> result;
    const std::array<char16_t, 256>& codepoints = bytelevel_codepoints();
    for (int byte = 0; byte < 256; ++byte) {
        result.emplace(codepoints[byte], static_cast<uint8_t>(byte));
    }
    return result;
}
std::string byte_token(const std::string& token, const std::map<char16_t, uint8_t>& decoder) {
    std::string result;
    for (char16_t code : utf16(token)) {
        auto found = decoder.find(code);
        require(found != decoder.end(), "BPE vocabulary contains a non-ByteLevel character");
        result.push_back(static_cast<char>(found->second));
    }
    require(!result.empty(), "empty BPE token");
    return result;
}
std::string token_content(const Json& config, const std::string& key) {
    if (!config.contains(key) || config.at(key).is_null()) return {};
    const Json& token = config.at(key);
    return token.is_object() ? token.at("content").string() : token.string();
}
}

HuggingFaceTokenizer::HuggingFaceTokenizer(const std::string& directory) {
    std::filesystem::path root = std::filesystem::path(directory);
    Json definition = Json::read((root / "tokenizer.json").string());
    Json config = Json::read((root / "tokenizer_config.json").string());
    require(definition.at("version").string() == "1.0", "unsupported tokenizer format version");
    require(definition.at("normalizer").at("type").string() == "NFC", "only NFC normalization is supported");
    const Json& pre = definition.at("pre_tokenizer");
    require(pre.at("type").string() == "Sequence" && pre.at("pretokenizers").array().size() == 2,
            "expected regex Split followed by ByteLevel");
    const Json& split = pre.at("pretokenizers").at(0);
    const Json& byte_level = pre.at("pretokenizers").at(1);
    require(split.at("type").string() == "Split" && split.at("behavior").string() == "Isolated" &&
            !split.bool_or("invert", false), "unsupported Split behavior");
    require(byte_level.at("type").string() == "ByteLevel" && !byte_level.bool_or("add_prefix_space", true) &&
            !byte_level.bool_or("use_regex", true), "unsupported ByteLevel preprocessing");
    require(definition.at("post_processor").at("type").string() == "ByteLevel" &&
            definition.at("decoder").at("type").string() == "ByteLevel", "unsupported postprocessor or decoder");
    const std::string pattern = split.at("pattern").at("Regex").string();
    const std::string qwen_pattern = R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
    require(pattern == qwen_pattern, "unsupported pretokenizer expression");
    splitter_ = std::make_unique<UnicodeRegexSplit>(pattern);
    const Json& model = definition.at("model");
    require(model.at("type").string() == "BPE" && model.at("dropout").is_null() && model.at("unk_token").is_null() &&
            model.string_or("continuing_subword_prefix", "").empty() && model.string_or("end_of_word_suffix", "").empty() &&
            !model.bool_or("byte_fallback", false) && !model.bool_or("fuse_unk", false), "unsupported BPE options");
    const Json::Object& vocabulary = model.at("vocab").object();
    require(vocabulary.size() >= 256 && vocabulary.size() < size_t(std::numeric_limits<int32_t>::max()), "invalid vocabulary size");
    bpe_.vocabulary_.resize(vocabulary.size());
    const std::map<char16_t, uint8_t> decoder = byte_decoder();
    std::map<std::string, int32_t> ids;
    for (const std::pair<const std::string, Json>& token : vocabulary) {
        int id = token.second.integer();
        require(id >= 0 && size_t(id) < vocabulary.size() && bpe_.vocabulary_[id].empty(), "invalid or duplicate vocabulary ID");
        bpe_.vocabulary_[id] = byte_token(token.first, decoder);
        ids.emplace(token.first, id);
    }
    for (const std::pair<const char16_t, unsigned char>& item : decoder) {
        std::string token = utf8(std::u16string(1, item.first));
        auto found = ids.find(token);
        require(found != ids.end(), "incomplete byte alphabet");
        bpe_.byte_ids_[item.second] = found->second;
    }
    for (const Json& item : model.at("merges").array()) {
        std::string left, right;
        if (item.is_array()) {
            require(item.array().size() == 2, "invalid BPE merge pair");
            left = item.at(0).string();
            right = item.at(1).string();
        } else {
            const std::string& merge = item.string();
            const size_t separator = merge.find(' ');
            require(separator != std::string::npos && separator != 0 && separator + 1 < merge.size() &&
                    merge.find(' ', separator + 1) == std::string::npos, "invalid BPE merge");
            left = merge.substr(0, separator);
            right = merge.substr(separator + 1);
        }
        auto a = ids.find(left), b = ids.find(right), output = ids.find(left + right);
        require(a != ids.end() && b != ids.end() && output != ids.end(), "merge token is absent from the vocabulary");
        require(bpe_.ranks_.emplace(pair_key(a->second, b->second), bpe_.merges_.size()).second, "duplicate merge pair");
        bpe_.merges_.push_back({a->second, b->second, output->second});
    }
    decoded_ = bpe_.vocabulary_;
    special_.resize(decoded_.size(), false);
    std::set<int32_t> added_ids;
    std::map<std::string, int32_t> added_text;
    for (const Json& item : definition.at("added_tokens").array()) {
        int id = item.at("id").integer();
        std::string text = item.at("content").string();
        require(id >= 0 && size_t(id) <= decoded_.size() && !text.empty() && added_ids.insert(id).second &&
                added_text.emplace(text, id).second, "invalid or duplicate added token");
        require(!item.bool_or("single_word", false) && !item.bool_or("lstrip", false) &&
                !item.bool_or("rstrip", false) && !item.bool_or("normalized", false), "unsupported added-token matching rules");
        bool special = item.at("special").boolean();
        if (size_t(id) == decoded_.size()) { decoded_.push_back(text); special_.push_back(special); }
        else { decoded_[id] = text; special_[id] = special; }
        added_.push_back({id, std::move(text), special});
    }
    LocalFunction special_id{[&](const std::string& key) {
        std::string text = token_content(config, key);
        if (text.empty()) return int32_t(-1);
        auto found = added_text.find(text);
        require(found != added_text.end(), "configured special token is absent from added_tokens: " + key);
        special_[found->second] = true;
        return found->second;
    }};
    bos_ = special_id("bos_token"); eos_ = special_id("eos_token"); pad_ = special_id("pad_token");
    add_bos_ = config.bool_or("add_bos_token", false); add_eos_ = config.bool_or("add_eos_token", false);
    require((!add_bos_ || bos_ >= 0) && (!add_eos_ || eos_ >= 0), "missing configured BOS/EOS token");
    bos_token_ = token_content(config, "bos_token");
    chat_template_ = config.string_or("chat_template", "");
    require(!config.bool_or("clean_up_tokenization_spaces", false), "tokenization cleanup is unsupported");
}

void HuggingFaceTokenizer::ordinary(const std::string& text, std::vector<int32_t>& result) const {
    if (text.empty()) return;
    for (const std::string& piece : splitter_->split(normalize_nfc(text))) {
        std::vector<int32_t> ids = bpe_.encode(piece, false);
        result.insert(result.end(), ids.begin(), ids.end());
    }
}

std::vector<int32_t> HuggingFaceTokenizer::encode(const std::string& text, bool add_special_tokens) const {
    std::vector<int32_t> result;
    if (add_special_tokens && add_bos_) result.push_back(bos_);
    size_t begin = 0, position = 0;
    while (position < text.size()) {
        const AddedToken* match = nullptr;
        for (const HuggingFaceTokenizer::AddedToken& token : added_)
            if (text.compare(position, token.text.size(), token.text) == 0 && (!match || token.text.size() > match->text.size()))
                match = &token;
        if (!match) { ++position; continue; }
        ordinary(text.substr(begin, position - begin), result);
        result.push_back(match->id);
        position += match->text.size();
        begin = position;
    }
    ordinary(text.substr(begin), result);
    if (add_special_tokens && add_eos_) result.push_back(eos_);
    return result;
}

std::string HuggingFaceTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special_tokens) const {
    std::string bytes;
    for (int32_t id : ids) {
        require(id >= 0, "negative token ID");
        // Tokenizers filters IDs without a vocabulary entry during decoding.
        if (size_t(id) >= decoded_.size()) continue;
        if (!skip_special_tokens || !special_[id]) bytes += decoded_[id];
    }
    return replace_invalid_utf8(bytes);
}

} // namespace elman
