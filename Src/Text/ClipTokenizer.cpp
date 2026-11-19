#include "Text/ClipTokenizer.h"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace elman {
namespace {

std::string token(const Json& config, const std::string& name) {
  const auto& entry = config.at(name);
  return entry.is_object() ? entry.at("content").string() : entry.string();
}

}  // namespace

ClipTokenizer::ClipTokenizer(const std::string& directory)
    : splitter_(R"(<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+)") {
  const auto root = std::filesystem::path(directory);
  const auto config = Json::read((root / "tokenizer_config.json").string());
  const auto vocab = Json::read((root / "vocab.json").string());
  if (config.string_or("tokenizer_class", "CLIPTokenizer") != "CLIPTokenizer" ||
      !config.bool_or("do_lower_case", true) || config.bool_or("add_prefix_space", false))
    throw std::invalid_argument("Unsupported CLIP tokenizer configuration");
  length_ = config.int_or("model_max_length", 77);
  if (length_ < 2 || vocab.object().size() > size_t(std::numeric_limits<int>::max()))
    throw std::invalid_argument("Invalid CLIP text length or vocabulary size");
  vocabulary_ = static_cast<int>(vocab.object().size());
  std::vector<bool> seen(vocabulary_, false);
  for (const auto& entry : vocab.object()) {
    const int id = entry.second.integer();
    if (id < 0 || id >= vocabulary_ || seen[id]) throw std::invalid_argument("Invalid CLIP vocabulary ID");
    seen[id] = true;
  }
  auto id = [&](const std::string& symbol) { return vocab.at(symbol).integer(); };
  bos_token_ = token(config, "bos_token");
  eos_token_ = token(config, "eos_token");
  if (bos_token_ != "<|startoftext|>" || eos_token_ != "<|endoftext|>")
    throw std::invalid_argument("Unsupported CLIP BOS or EOS token");
  bos_ = id(bos_token_);
  eos_ = id(eos_token_);
  pad_ = id(token(config, "pad_token"));
  const auto& codepoints = bytelevel_codepoints();
  for (size_t byte = 0; byte < codepoints.size(); ++byte) {
    const auto symbol = utf8(std::u16string(1, codepoints[byte]));
    bpe_.byte_ids_[byte] = id(symbol);
    endings_[byte] = id(symbol + "</w>");
  }
  std::ifstream merges(root / "merges.txt");
  if (!merges) throw std::runtime_error("Cannot open CLIP merges.txt");
  std::string line;
  while (std::getline(merges, line)) {
    if (line.empty() || line.front() == '#') continue;
    std::string left, right, extra;
    std::istringstream fields(line);
    if (!(fields >> left >> right) || fields >> extra) throw std::invalid_argument("Invalid CLIP merge line");
    const int32_t a = id(left), b = id(right);
    const uint64_t key = (uint64_t(uint32_t(a)) << 32) | uint32_t(b);
    if (!bpe_.ranks_.emplace(key, bpe_.merges_.size()).second) throw std::invalid_argument("Duplicate CLIP merge");
    bpe_.merges_.push_back({a, b, id(left + right)});
  }
  if (!merges.eof()) throw std::runtime_error("Cannot read CLIP merges.txt");
}

std::vector<int32_t> ClipTokenizer::encode(const std::string& prompt) const {
  std::vector<int32_t> ids{bos_};
  for (const auto& word : splitter_.split(lowercase_unicode(normalize_nfc(prompt)), false)) {
    if (ids.size() >= size_t(length_ - 1)) break;
    if (word == bos_token_) ids.push_back(bos_);
    else if (word == eos_token_) ids.push_back(eos_);
    else bpe_.encode_bytes(word, 0, word.size(), ids, &endings_);
  }
  if (ids.size() >= size_t(length_)) ids.resize(size_t(length_ - 1));
  ids.push_back(eos_);
  ids.resize(length_, pad_);
  return ids;
}

}  // namespace elman
