#pragma once

#include <array>
#include <stdexcept>
#include <string>
#include <vector>
#include "IO/Json.h"

namespace elman {

// A fixed ASCII character vocabulary travels with the model/checkpoint.
class CharacterTokenizer {
 public:
  explicit CharacterTokenizer(const Json& config) : characters_(config.at("characters").string()) {
    if (config.at("type").string() != "character" || characters_.size() < 2)
      throw std::invalid_argument("Invalid character tokenizer");
    ids_.fill(-1);
    for (size_t i = 0; i < characters_.size(); ++i) {
      const auto c = static_cast<unsigned char>(characters_[i]);
      if (c >= 128 || ids_[c] >= 0) throw std::invalid_argument("Vocabulary must contain unique ASCII characters");
      ids_[c] = int32_t(i);
    }
  }
  size_t size() const { return characters_.size(); }
  std::vector<int32_t> encode(const std::string& text) const {
    std::vector<int32_t> result;
    result.reserve(text.size());
    for (unsigned char c : text) {
      if (ids_[c] < 0) throw std::invalid_argument("Text contains a character outside the model vocabulary: " + std::to_string(c));
      result.push_back(ids_[c]);
    }
    return result;
  }
  char decode(int32_t token) const {
    if (token < 0 || size_t(token) >= characters_.size()) throw std::invalid_argument("Character token outside vocabulary");
    return characters_[token];
  }
 private:
  std::string characters_;
  std::array<int32_t, 256> ids_;
};

}  // namespace elman
