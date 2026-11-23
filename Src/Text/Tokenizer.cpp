#include "Text/Tokenizer.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Tokenizer: " + message);
}

uint64_t pair_key(int32_t left, int32_t right) {
    return (uint64_t(uint32_t(left)) << 32) | uint32_t(right);
}

std::string hex(const std::string& bytes) {
    const char* digits = "0123456789abcdef";
    std::string result;
    result.reserve(bytes.size() * 2);
    for (unsigned char byte : bytes) {
        result.push_back(digits[byte >> 4]);
        result.push_back(digits[byte & 15]);
    }
    return result;
}

std::string unhex(const std::string& value) {
    require(value.size() % 2 == 0, "invalid hexadecimal vocabulary entry");
    int (*const nibble)(char) = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        throw std::invalid_argument("Tokenizer: invalid hexadecimal digit");
    };
    std::string result;
    for (size_t i = 0; i < value.size(); i += 2)
        result.push_back(char(nibble(value[i]) * 16 + nibble(value[i + 1])));
    return result;
}

std::string read_bytes(const std::filesystem::path& path, size_t limit = size_t(-1)) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    require(bool(stream), "cannot open input: " + path.string());
    std::streampos end = stream.tellg();
    require(end >= 0 && uint64_t(end) <= uint64_t(limit), "input exceeds corpus byte limit");
    std::string text(size_t(end), '\0');
    stream.seekg(0);
    if (!text.empty()) stream.read(&text[0], std::streamsize(text.size()));
    require(bool(stream), "failed to read input: " + path.string());
    return text;
}

void create_parent(const std::filesystem::path& path) {
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
}

} // namespace

bool BytePairTokenizer::is_special(int32_t id) const {
    return std::find(special_ids_.begin(), special_ids_.end(), id) != special_ids_.end();
}

BytePairTokenizer BytePairTokenizer::train(const std::vector<std::string>& documents,
                                           const TokenizerTrainingOptions& options) {
    require(options.special_tokens.size() <= size_t(std::numeric_limits<int>::max() - 256), "too many special tokens");
    require(options.vocabulary_size >= 256 + int(options.special_tokens.size()), "vocabulary must contain all 256 bytes and special tokens");
    require(options.minimum_frequency > 0, "minimum frequency must be positive");
    require(options.maximum_token_bytes > 0, "maximum token size must be positive");
    BytePairTokenizer model;
    for (int i = 0; i < 256; ++i) model.vocabulary_.emplace_back(1, char(i));
    std::set<std::string> unique_specials;
    for (const std::string& token : options.special_tokens) {
        require(!token.empty() && unique_specials.insert(token).second, "special tokens must be nonempty and unique");
        model.special_ids_.push_back(model.vocabulary_size());
        model.vocabulary_.push_back(token);
    }
    std::vector<std::vector<int32_t>> sequences;
    for (const std::string& document : documents) sequences.push_back(model.encode(document));
    std::unordered_map<std::string, int32_t> ordinary_ids;
    for (int32_t i = 0; i < 256; ++i) ordinary_ids.emplace(model.vocabulary_[i], i);
    while (model.vocabulary_size() < options.vocabulary_size) {
        // Ordered keys make frequency ties deterministic across runs/platforms.
        std::map<uint64_t, uint64_t> frequencies;
        for (const std::vector<int>& sequence : sequences) {
            for (size_t i = 1; i < sequence.size(); ++i) {
                int32_t left = sequence[i - 1], right = sequence[i];
                if (!model.is_special(left) && !model.is_special(right) &&
                    model.vocabulary_[left].size() + model.vocabulary_[right].size() <= options.maximum_token_bytes)
                    ++frequencies[pair_key(left, right)];
            }
        }
        uint64_t best_pair = 0, best_count = 0;
        for (const std::pair<const uint64_t, uint64_t>& entry : frequencies) {
            if (entry.second > best_count) {
                best_pair = entry.first;
                best_count = entry.second;
            }
        }
        if (best_count < options.minimum_frequency) break;
        int32_t left = int32_t(best_pair >> 32), right = int32_t(uint32_t(best_pair));
        std::string bytes = model.vocabulary_[left] + model.vocabulary_[right];
        auto existing = ordinary_ids.find(bytes);
        int32_t result = existing == ordinary_ids.end() ? model.vocabulary_size() : existing->second;
        if (existing == ordinary_ids.end()) {
            ordinary_ids.emplace(bytes, result);
            model.vocabulary_.push_back(std::move(bytes));
        }
        require(model.ranks_.emplace(best_pair, model.merges_.size()).second, "duplicate training merge");
        model.merges_.push_back({left, right, result});
        for (std::vector<int>& sequence : sequences) {
            size_t target = 0;
            for (size_t i = 0; i < sequence.size(); ++i) {
                if (sequence[i] == left && i + 1 < sequence.size() && sequence[i + 1] == right) {
                    sequence[target++] = result;
                    ++i;
                } else sequence[target++] = sequence[i];
            }
            sequence.resize(target);
        }
    }
    return model;
}

void BytePairTokenizer::encode_bytes(const std::string& text, size_t begin, size_t end,
                                     std::vector<int32_t>& output, const std::array<int32_t, 256>* endings) const {
    if (begin == end) return;
    require(end - begin <= size_t(std::numeric_limits<int>::max()), "input is too long");
    struct Node { int32_t token; int previous, next; bool alive = true; };
    struct Candidate {
        size_t rank;
        int left, right;
        bool operator<(const Candidate& other) const {
            return rank != other.rank ? rank > other.rank : left > other.left;
        }
    };
    std::vector<Node> nodes;
    nodes.reserve(end - begin);
    int count = int(end - begin);
    for (int i = 0; i < count; ++i) {
        const bool last = i + 1 == count;
        const std::array<int32_t, 256>& symbols = last && endings ? *endings : byte_ids_;
        nodes.push_back({symbols[uint8_t(text[begin + i])], i - 1, last ? -1 : i + 1});
    }
    std::priority_queue<Candidate> queue;
    LocalFunction enqueue{[&](int left) {
        if (left < 0 || !nodes[left].alive || nodes[left].next < 0) return;
        int right = nodes[left].next;
        auto found = ranks_.find(pair_key(nodes[left].token, nodes[right].token));
        if (found != ranks_.end()) queue.push({found->second, left, right});
    }};
    for (int i = 0; i < count; ++i) enqueue(i);
    while (!queue.empty()) {
        Candidate candidate = queue.top();
        queue.pop();
        Node& left = nodes[candidate.left];
        Node& right = nodes[candidate.right];
        const BytePairTokenizer::Merge& merge = merges_[candidate.rank];
        if (!left.alive || !right.alive || left.next != candidate.right ||
            left.token != merge.left || right.token != merge.right) continue;
        left.token = merge.result;
        left.next = right.next;
        right.alive = false;
        if (right.next >= 0) nodes[right.next].previous = candidate.left;
        enqueue(left.previous);
        enqueue(candidate.left);
    }
    for (int i = 0; i >= 0; i = nodes[i].next) output.push_back(nodes[i].token);
}

std::vector<int32_t> BytePairTokenizer::encode(const std::string& text, bool recognize_special) const {
    require(vocabulary_.size() >= 256, "model is not initialized");
    std::vector<int32_t> result;
    size_t ordinary_begin = 0, position = 0;
    while (recognize_special && position < text.size() && !special_ids_.empty()) {
        int32_t match = -1;
        for (int32_t id : special_ids_) {
            const std::string& token = vocabulary_[id];
            if (text.compare(position, token.size(), token) == 0 &&
                (match < 0 || token.size() > vocabulary_[match].size())) match = id;
        }
        if (match < 0) { ++position; continue; }
        encode_bytes(text, ordinary_begin, position, result);
        result.push_back(match);
        position += vocabulary_[match].size();
        ordinary_begin = position;
    }
    encode_bytes(text, ordinary_begin, text.size(), result);
    return result;
}

std::string BytePairTokenizer::decode(const std::vector<int32_t>& ids, bool skip_special) const {
    std::string result;
    for (int32_t id : ids) {
        require(id >= 0 && size_t(id) < vocabulary_.size(), "token ID is outside the vocabulary");
        if (!skip_special || !is_special(id)) result += vocabulary_[id];
    }
    return result;
}

Json BytePairTokenizer::to_json() const {
    require(vocabulary_.size() >= 256, "model is not initialized");
    Json::Array vocabulary, specials, merges;
    for (const std::string& token : vocabulary_) vocabulary.emplace_back(hex(token));
    for (int32_t id : special_ids_) specials.emplace_back(int(id));
    for (const BytePairTokenizer::Merge& merge : merges_)
        merges.emplace_back(Json::Array{int(merge.left), int(merge.right), int(merge.result)});
    return Json::Object{{"format", "elman.tokenizer"}, {"version", 1}, {"model", "byte_bpe"},
                        {"normalization", "none"}, {"pretokenizer", "none"},
                        {"vocabulary_hex", std::move(vocabulary)}, {"special_ids", std::move(specials)},
                        {"merges", std::move(merges)}};
}

BytePairTokenizer BytePairTokenizer::from_json(const Json& definition) {
    require(definition.at("format").string() == "elman.tokenizer" && definition.at("version").integer() == 1,
            "unsupported tokenizer format/version");
    require(definition.at("model").string() == "byte_bpe" && definition.at("normalization").string() == "none" &&
            definition.at("pretokenizer").string() == "none", "unsupported tokenizer pipeline");
    BytePairTokenizer model;
    const Json::Array& vocabulary = definition.at("vocabulary_hex").array();
    require(vocabulary.size() >= 256 && vocabulary.size() <= size_t(std::numeric_limits<int>::max()), "invalid vocabulary size");
    for (const Json& value : vocabulary) {
        std::string token = unhex(value.string());
        require(!token.empty(), "empty vocabulary token");
        model.vocabulary_.push_back(std::move(token));
    }
    std::vector<bool> introduced(vocabulary.size(), false);
    for (int i = 0; i < 256; ++i) {
        require(model.vocabulary_[i] == std::string(1, char(i)), "byte alphabet does not match canonical IDs");
        introduced[i] = true;
    }
    std::set<std::string> special_texts;
    for (const Json& entry : definition.at("special_ids").array()) {
        int id = entry.integer();
        require(id >= 256 && size_t(id) < vocabulary.size() && !introduced[id], "invalid or duplicate special ID");
        require(special_texts.insert(model.vocabulary_[id]).second, "duplicate special token text");
        introduced[id] = true;
        model.special_ids_.push_back(id);
    }
    std::set<std::string> ordinary_texts;
    for (size_t i = 0; i < vocabulary.size(); ++i)
        if (!model.is_special(int32_t(i))) require(ordinary_texts.insert(model.vocabulary_[i]).second, "duplicate ordinary token text");
    for (const Json& item : definition.at("merges").array()) {
        require(item.array().size() == 3, "merge must contain two inputs and one output");
        Merge merge{item.at(0).integer(), item.at(1).integer(), item.at(2).integer()};
        for (int id : {merge.left, merge.right, merge.result})
            require(id >= 0 && size_t(id) < vocabulary.size() && !model.is_special(id), "invalid merge ID");
        require(introduced[merge.left] && introduced[merge.right], "merge references an unavailable token");
        require(model.vocabulary_[merge.result] == model.vocabulary_[merge.left] + model.vocabulary_[merge.right], "merge bytes do not match output");
        require(model.ranks_.emplace(pair_key(merge.left, merge.right), model.merges_.size()).second, "duplicate merge pair");
        introduced[merge.result] = true;
        model.merges_.push_back(merge);
    }
    require(std::all_of(introduced.begin(), introduced.end(), [](bool value) { return value; }), "unreachable vocabulary entry");
    return model;
}

BytePairTokenizer BytePairTokenizer::load(const std::string& path) { return from_json(Json::read(path)); }
void BytePairTokenizer::save(const std::string& path) const { to_json().write(path); }

int tokenizer_command(const std::string& command, int argc, char** argv) {
    require(argc >= 3, "a model or training configuration is required");
    if (command == "tokenizer-train") {
        require(argc == 3, "tokenizer-train takes one configuration file");
        Json definition = Json::read(argv[2]);
        std::filesystem::path root = std::filesystem::absolute(argv[2]).parent_path();
        TokenizerTrainingOptions options;
        options.vocabulary_size = definition.int_or("vocabulary_size", 1024);
        int frequency = definition.int_or("minimum_frequency", 2);
        int maximum_token_bytes = definition.int_or("maximum_token_bytes", 128);
        int maximum_corpus_bytes = definition.int_or("maximum_corpus_bytes", 32 * 1024 * 1024);
        require(frequency > 0 && maximum_token_bytes > 0 && maximum_corpus_bytes > 0, "training limits must be positive");
        options.minimum_frequency = uint64_t(frequency);
        options.maximum_token_bytes = size_t(maximum_token_bytes);
        if (definition.contains("special_tokens"))
            for (const Json& item : definition.at("special_tokens").array()) options.special_tokens.push_back(item.string());
        size_t remaining = size_t(maximum_corpus_bytes);
        std::vector<std::string> documents;
        for (const Json& item : definition.at("corpus").array()) {
            documents.push_back(read_bytes(root / item.string(), remaining));
            remaining -= documents.back().size();
        }
        require(!documents.empty(), "at least one corpus file is required");
        BytePairTokenizer model = BytePairTokenizer::train(documents, options);
        std::filesystem::path output = root / definition.at("output").string();
        create_parent(output);
        model.save(output.string());
        std::cout << Json(Json::Object{{"vocabulary_size", model.vocabulary_size()},
                                      {"merges", double(model.merge_count())}, {"output", output.string()}}).dump() << '\n';
        return 0;
    }
    require(command == "tokenizer-encode" || command == "tokenizer-decode", "unknown tokenizer command");
    std::string input, output;
    for (int i = 3; i < argc; i += 2) {
        require(i + 1 < argc, "missing option value");
        std::string key = argv[i];
        if (key == "--input") { require(input.empty(), "duplicate input option"); input = argv[i + 1]; }
        else if (key == "--output") { require(output.empty(), "duplicate output option"); output = argv[i + 1]; }
        else throw std::invalid_argument("Tokenizer: unknown option " + key);
    }
    require(!input.empty() && !output.empty(), "input and output paths are required");
    BytePairTokenizer model = BytePairTokenizer::load(argv[2]);
    create_parent(output);
    if (command == "tokenizer-encode") {
        Json::Array ids;
        for (int id : model.encode(read_bytes(input))) ids.emplace_back(int(id));
        Json(std::move(ids)).write(output);
    } else {
        Json definition = Json::read(input);
        std::vector<int32_t> ids;
        for (const Json& id : definition.array()) ids.push_back(id.integer());
        std::string text = model.decode(ids);
        std::ofstream stream(output, std::ios::binary);
        stream.write(text.data(), std::streamsize(text.size()));
        require(bool(stream), "failed to write decoded output");
    }
    return 0;
}

} // namespace elman
