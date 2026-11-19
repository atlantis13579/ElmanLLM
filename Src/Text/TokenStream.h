#pragma once

#include "Text/HuggingFaceTokenizer.h"

namespace elman {

// The tokenizer must outlive its stream. Incomplete UTF-8 prefixes are held
// until resolved; finish() emits the same replacement characters as full decode.
class TokenizerStream {
public:
    explicit TokenizerStream(const HuggingFaceTokenizer& tokenizer, bool skip_special_tokens = false)
        : tokenizer_(tokenizer), skip_special_(skip_special_tokens) {}
    std::string push(int32_t id);
    std::string finish();
private:
    const HuggingFaceTokenizer& tokenizer_;
    bool skip_special_, finished_ = false;
    std::string pending_;
};

// Stops are matched only in generated text. Potential stop prefixes are held
// so a streaming consumer never observes a stop string that must be retracted.
class TextStopFilter {
public:
    explicit TextStopFilter(std::vector<std::string> stops = {});
    std::string push(const std::string& text);
    std::string finish();
    bool stopped() const { return stopped_; }
    const std::string& matched_stop() const { return matched_; }
private:
    std::vector<std::string> stops_;
    std::string pending_, matched_;
    bool stopped_ = false, finished_ = false;
};

} // namespace elman
