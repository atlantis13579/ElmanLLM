#pragma once

#include "Text/Tokenizer.h"
#include "Text/Unicode.h"

namespace elman {

class ClipTokenizer {
 public:
  explicit ClipTokenizer(const std::string& directory);
  std::vector<int32_t> encode(const std::string& prompt) const;
  int length() const { return length_; }
  int vocabulary_size() const { return vocabulary_; }

 private:
  BytePairTokenizer bpe_;
  UnicodeRegexSplit splitter_;
  std::array<int32_t, 256> endings_{};
  std::string bos_token_, eos_token_;
  int32_t bos_, eos_, pad_;
  int length_, vocabulary_;
};

}  // namespace elman
