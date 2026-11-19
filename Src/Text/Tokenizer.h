#pragma once

#include <cstdint>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "IO/Json.h"

namespace elman {

struct TokenizerTrainingOptions {
    int vocabulary_size = 1024;
    uint64_t minimum_frequency = 2;
    size_t maximum_token_bytes = 128;
    std::vector<std::string> special_tokens;
};

// A lossless byte BPE model. Training does not modify neural model weights.
class BytePairTokenizer {
public:
    static BytePairTokenizer train(const std::vector<std::string>& documents,
                                   const TokenizerTrainingOptions& options = {});
    static BytePairTokenizer from_json(const Json& definition);
    static BytePairTokenizer load(const std::string& path);
    Json to_json() const;
    void save(const std::string& path) const;
    std::vector<int32_t> encode(const std::string& text, bool recognize_special = true) const;
    std::string decode(const std::vector<int32_t>& ids, bool skip_special = false) const;
    int vocabulary_size() const { return static_cast<int>(vocabulary_.size()); }
    size_t merge_count() const { return merges_.size(); }

private:
    friend class HuggingFaceTokenizer;
    friend class ClipTokenizer;
    struct Merge { int32_t left, right, result; };
    std::vector<std::string> vocabulary_;
    std::vector<Merge> merges_;
    std::vector<int32_t> special_ids_;
    std::unordered_map<uint64_t, size_t> ranks_;
    std::array<int32_t, 256> byte_ids_ = [] {
        std::array<int32_t, 256> result{};
        for (int i = 0; i < 256; ++i) result[i] = i;
        return result;
    }();
    bool is_special(int32_t id) const;
    void encode_bytes(const std::string& text, size_t begin, size_t end,
                      std::vector<int32_t>& output, const std::array<int32_t, 256>* endings = nullptr) const;
};

int tokenizer_command(const std::string& command, int argc, char** argv);

} // namespace elman
