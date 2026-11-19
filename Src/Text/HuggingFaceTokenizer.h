#pragma once

#include "Text/Tokenizer.h"
#include "Text/Unicode.h"

namespace elman {

// Imports the NFC/isolated-regex/ByteLevel BPE pipeline used by the pinned
// DeepSeek/Qwen tokenizer. Neural weights are not involved in tokenization.
class HuggingFaceTokenizer {
public:
    explicit HuggingFaceTokenizer(const std::string& model_directory);
    std::vector<int32_t> encode(const std::string& text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special_tokens = false) const;
    // Supports the pinned DeepSeek template. Other Jinja templates are rejected.
    std::string render_chat(const Json::Array& messages, bool add_generation_prompt = true) const;
    std::vector<int32_t> encode_chat(const Json::Array& messages, bool add_generation_prompt = true) const;
    int32_t bos_id() const { return bos_; }
    int32_t eos_id() const { return eos_; }
    int32_t pad_id() const { return pad_; }
    size_t vocabulary_size() const { return decoded_.size(); }
    const std::string& chat_template() const { return chat_template_; }
    const std::string& bos_token() const { return bos_token_; }
private:
    friend class TokenizerStream;
    struct AddedToken { int32_t id; std::string text; bool special; };
    BytePairTokenizer bpe_;
    std::vector<std::string> decoded_;
    std::vector<bool> special_;
    std::vector<AddedToken> added_;
    std::unique_ptr<UnicodeRegexSplit> splitter_;
    bool add_bos_ = false, add_eos_ = false;
    int32_t bos_ = -1, eos_ = -1, pad_ = -1;
    std::string bos_token_, chat_template_;
    void ordinary(const std::string& text, std::vector<int32_t>& result) const;
};

} // namespace elman
