#include "Text/Tokenizer.h"

#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool condition, const char* message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "invalid tokenizer input was accepted");
}
}

int main() {
    try {
        using namespace elman;
        TokenizerTrainingOptions options;
        options.vocabulary_size = 310;
        options.special_tokens = {"<end>", "<end>long", "<begin>"};
        std::vector<std::string> corpus = {
            "banana banana banana banana\n",
            "hello world! hello world! hello world!\n",
            u8"\u4f60\u597d\u4e16\u754c \u4f60\u597d\u4e16\u754c \U0001f642\U0001f642",
            "<begin>hello world!<end><begin>banana banana<end>"};
        auto model = BytePairTokenizer::train(corpus, options);
        check(model.vocabulary_size() > 259 && model.vocabulary_size() <= 310, "training vocabulary size");
        check(model.merge_count() > 0, "no BPE merges were learned");
        check(model.encode(corpus[0]).size() < corpus[0].size() / 2, "training did not compress repeated text");
        auto loaded = BytePairTokenizer::from_json(Json::parse(model.to_json().dump()));
        check(model.to_json().dump() == loaded.to_json().dump(), "serialization changed the model");
        auto repeated = BytePairTokenizer::train(corpus, options);
        check(model.to_json().dump() == repeated.to_json().dump(), "training is not deterministic");
        for (const auto& document : corpus) {
            auto ids = model.encode(document);
            check(model.decode(ids) == document, "training text did not round trip");
            check(ids == loaded.encode(document), "reload changed token IDs");
        }
        check(model.encode("").empty(), "empty text should have no tokens");
        check(model.decode({}).empty(), "empty tokens should decode to empty text");
        check(model.encode("<end>long") == std::vector<int32_t>{257}, "special token matching must prefer longest text");
        check(model.encode("<end>") == std::vector<int32_t>{256}, "special token ID changed");
        check(model.decode(model.encode("x<end>y"), true) == "xy", "skip special tokens");
        check(model.decode(model.encode("<end>", false), true) == "<end>", "ordinary text was treated as a special token");
        std::string alphabet;
        for (int i = 0; i < 256; ++i) alphabet.push_back(char(i));
        check(loaded.decode(loaded.encode(alphabet)) == alphabet, "all byte values must round trip");
        std::mt19937 generator(731);
        for (size_t size : {size_t(1), size_t(2), size_t(257), size_t(4096)}) {
            std::string text(size, '\0');
            for (char& byte : text) byte = char(generator() & 255);
            check(loaded.decode(loaded.encode(text)) == text, "arbitrary bytes did not round trip");
        }
        // Overlapping pairs and merge-rank priority must produce stable IDs.
        TokenizerTrainingOptions overlap;
        overlap.vocabulary_size = 280;
        overlap.minimum_frequency = 1;
        auto overlapping = BytePairTokenizer::train({"aaaaaaaaaaaaaaaaa", "abababababab", "abcabcabcabc"}, overlap);
        for (const auto& text : {"a", "aa", "aaa", "aaaaaaa", "abababa", "abcabcab"})
            check(overlapping.decode(overlapping.encode(text)) == text, "overlap round trip");
        auto empty = BytePairTokenizer::train({""});
        check(empty.vocabulary_size() == 256 && empty.merge_count() == 0, "empty training corpus");
        rejects([&] { model.decode({-1}); });
        rejects([&] { model.decode({model.vocabulary_size()}); });
        rejects([&] { BytePairTokenizer().encode("x"); });
        auto invalid = options;
        invalid.vocabulary_size = 255;
        rejects([&] { BytePairTokenizer::train(corpus, invalid); });
        invalid = options;
        invalid.special_tokens = {"x", "x"};
        rejects([&] { BytePairTokenizer::train(corpus, invalid); });
        invalid.special_tokens = {""};
        rejects([&] { BytePairTokenizer::train(corpus, invalid); });
        invalid = options;
        invalid.minimum_frequency = 0;
        rejects([&] { BytePairTokenizer::train(corpus, invalid); });
        auto object = model.to_json().object();
        auto merges = object.at("merges").array();
        merges.push_back(merges.front());
        object["merges"] = merges;
        rejects([&] { BytePairTokenizer::from_json(object); });
        object = model.to_json().object();
        auto vocabulary = object.at("vocabulary_hex").array();
        vocabulary[0] = "gg";
        object["vocabulary_hex"] = vocabulary;
        rejects([&] { BytePairTokenizer::from_json(object); });
        object = model.to_json().object();
        merges = object.at("merges").array();
        auto merge = merges[0].array();
        merge[2] = 0;
        merges[0] = merge;
        object["merges"] = merges;
        rejects([&] { BytePairTokenizer::from_json(object); });
        std::cout << "Tokenizer checks passed: " << checks << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
