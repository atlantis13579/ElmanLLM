#include "Text/Tokenizer.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Commands/Commands.h"

namespace elman {
namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument("Tokenizer: " + message);
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

}

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
