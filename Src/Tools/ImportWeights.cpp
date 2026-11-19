#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

#include "IO/AtomicFile.h"
#include "IO/Gguf.h"
#include "IO/Json.h"
#include "IO/TensorArchive.h"
#include "Models/LLM/Qwen2.h"

namespace {

using namespace elman;
namespace fs = std::filesystem;
using Names = std::map<std::string, std::string>;

void save_gguf(const GgufFile& file, const Names& names, const fs::path& output) {
    std::vector<SafetensorSpec> specs;
    for (const auto& [name, source] : names) {
        const auto& info = file.tensors().at(source);
        specs.push_back({name, info.dtype, info.shape});
    }
    write_file_atomically(output, [&](const fs::path& path) {
        std::ofstream stream(path, std::ios::binary);
        if (!stream) throw std::runtime_error("Cannot open Safetensors output");
        write_safetensors_header(stream, specs);
        for (const auto& [name, source] : names) {
            const auto data = file.read(source);
            if (data.size() > size_t(std::numeric_limits<std::streamsize>::max()))
                throw std::runtime_error("Safetensors payload is too large to write");
            stream.write(reinterpret_cast<const char*>(data.data()), std::streamsize(data.size()));
            if (!stream) throw std::runtime_error("Cannot write Safetensors payload " + name);
        }
        stream.close();
        if (!stream) throw std::runtime_error("Cannot close Safetensors output");
    });
}

double number(const GgufFile& file, const std::string& key, double fallback = 0) {
    const auto found = file.numbers().find(key);
    return found == file.numbers().end() ? fallback : found->second;
}

void save_pytorch_directory(const fs::path& input, const fs::path& output) {
    const auto source = fs::canonical(input);
    const auto index = source / "pytorch_model.bin.index.json";
    std::map<std::string, std::vector<std::string>> shards;
    if (fs::exists(index)) {
        const auto contents = Json::read(index);
        for (const auto& [tensor, entry] : contents.at("weight_map").object()) {
            const auto filename = entry.string();
            const auto path = fs::u8path(filename);
            if (path.has_root_path() || path.filename() != path || filename.find_first_of(":/\\") != std::string::npos ||
                model_file_extension(path) != ".bin" || fs::canonical(source / path).parent_path() != source)
                throw std::invalid_argument("Invalid PyTorch index shard " + filename);
            shards[filename].push_back(tensor);
        }
        if (shards.empty()) throw std::invalid_argument("PyTorch shard index is empty");
    }
    if (!fs::is_regular_file(source / "config.json"))
        throw std::invalid_argument("PyTorch model directory requires config.json");
    if (!fs::create_directories(output)) throw std::invalid_argument("Model output directory must be new");
    std::vector<fs::path> files;
    try {
        for (const auto* name : {"config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                                 "chat_template.jinja", "special_tokens_map.json"}) {
            if (!fs::is_regular_file(source / name)) continue;
            files.push_back(output / name);
            fs::copy_file(source / name, files.back());
        }
        if (shards.empty()) {
            files.push_back(output / "model.safetensors");
            save_safetensors(files.back(), load_tensor_archive(source / "pytorch_model.bin", Backend::CPU));
        } else {
            Json::Object map;
            size_t shard = 0;
            for (const auto& [filename, expected] : shards) {
                const auto tensors = load_tensor_archive(source / fs::u8path(filename), Backend::CPU);
                if (tensors.size() != expected.size())
                    throw std::invalid_argument("PyTorch index does not account for every tensor in " + filename);
                const auto name = "model-" + std::to_string(++shard) + ".safetensors";
                for (const auto& tensor : expected) {
                    if (!tensors.count(tensor)) throw std::invalid_argument("PyTorch shard is missing tensor " + tensor);
                    map.emplace(tensor, name);
                }
                files.push_back(output / name);
                save_safetensors(files.back(), tensors);
            }
            files.push_back(output / "model.safetensors.index.json");
            write_file_atomically(files.back(), [&](const fs::path& path) {
                std::ofstream stream(path, std::ios::binary);
                stream << Json(Json::Object{{"weight_map", std::move(map)}}).dump() << '\n';
                stream.close();
                if (!stream) throw std::runtime_error("Cannot write Safetensors shard index");
            });
        }
    } catch (...) {
        std::error_code ignored;
        for (const auto& file : files) fs::remove(file, ignored);
        fs::remove(output, ignored);
        throw;
    }
}

void save_qwen2(const GgufFile& file, const fs::path& directory, const fs::path& tokenizer) {
    const auto architecture = file.strings().find("general.architecture");
    const auto scaling = file.strings().find("qwen2.rope.scaling.type");
    if (architecture == file.strings().end() || architecture->second != "qwen2" ||
        (scaling != file.strings().end() && scaling->second != "none") ||
        number(file, "qwen2.rope.scaling.factor", 1) != 1 ||
        number(file, "qwen2.attention.sliding_window") != 0)
        throw std::invalid_argument("Qwen2 import requires ordinary Qwen2 attention and unscaled rotary embeddings");
    if (number(file, "split.count", 1) != 1)
        throw std::invalid_argument("Split GGUF checkpoints require merging before Qwen2 import");
    const auto& tensors = file.tensors();
    const auto& embedding = tensors.at("token_embd.weight").shape;
    if (embedding.size() != 2) throw std::invalid_argument("Invalid GGUF Qwen2 token embeddings");
    const auto heads = number(file, "qwen2.attention.head_count");
    const bool tied = !tensors.count("output.weight");
    Json::Object config{{"model_type", "qwen2"}, {"vocab_size", embedding.front()},
                        {"hidden_size", number(file, "qwen2.embedding_length")},
                        {"intermediate_size", number(file, "qwen2.feed_forward_length")},
                        {"num_hidden_layers", number(file, "qwen2.block_count")}, {"num_attention_heads", heads},
                        {"num_key_value_heads", number(file, "qwen2.attention.head_count_kv", heads)},
                        {"max_position_embeddings", number(file, "qwen2.context_length")},
                        {"rms_norm_eps", number(file, "qwen2.attention.layer_norm_rms_epsilon", 1e-6)},
                        {"rope_theta", number(file, "qwen2.rope.freq_base", 10000)}, {"tie_word_embeddings", tied}};
    if (file.numbers().count("tokenizer.ggml.eos_token_id"))
        config.emplace("eos_token_id", number(file, "tokenizer.ggml.eos_token_id"));
    const auto layout = Qwen2Config::from_json(config);
    if (tensors.size() != 2 + size_t(layout.layers) * 12 + size_t(!tied))
        throw std::invalid_argument("GGUF Qwen2 tensor count differs from the configuration");
    Names names;
    auto map = [&](const std::string& target, const std::string& source, const Shape& expected) {
        const auto found = tensors.find(source);
        if (found == tensors.end() || found->second.shape != expected || !is_compute_float(found->second.dtype))
            throw std::invalid_argument("Missing or incompatible Qwen2 GGUF tensor " + source);
        names.emplace(target, source);
    };
    map("model.embed_tokens.weight", "token_embd.weight", {layout.vocabulary, layout.hidden});
    map("model.norm.weight", "output_norm.weight", {layout.hidden});
    if (!tied) map("lm_head.weight", "output.weight", {layout.vocabulary, layout.hidden});
    for (int i = 0; i < layout.layers; ++i) {
        const auto source = "blk." + std::to_string(i) + '.';
        const auto target = "model.layers." + std::to_string(i) + '.';
        const auto kv_width = layout.head_dim() * layout.kv_heads;
        map(target + "input_layernorm.weight", source + "attn_norm.weight", {layout.hidden});
        map(target + "post_attention_layernorm.weight", source + "ffn_norm.weight", {layout.hidden});
        for (const auto* kind : {"q", "k", "v"}) {
            const auto width = *kind == 'q' ? layout.hidden : kv_width;
            const auto projection = target + "self_attn." + kind + "_proj.";
            const auto weight = source + "attn_" + kind + '.';
            map(projection + "weight", weight + "weight", {width, layout.hidden});
            map(projection + "bias", weight + "bias", {width});
        }
        map(target + "self_attn.o_proj.weight", source + "attn_output.weight", {layout.hidden, layout.hidden});
        map(target + "mlp.gate_proj.weight", source + "ffn_gate.weight", {layout.intermediate, layout.hidden});
        map(target + "mlp.up_proj.weight", source + "ffn_up.weight", {layout.intermediate, layout.hidden});
        map(target + "mlp.down_proj.weight", source + "ffn_down.weight", {layout.hidden, layout.intermediate});
    }
    if (!tokenizer.empty() && !fs::is_regular_file(tokenizer))
        throw std::invalid_argument("Qwen2 tokenizer must be an existing tokenizer.json file");
    if (!fs::create_directories(directory)) throw std::invalid_argument("Qwen2 output directory must be new");
    try {
        save_gguf(file, names, directory / "model.safetensors");
        if (!tokenizer.empty()) fs::copy_file(tokenizer, directory / "tokenizer.json");
        write_file_atomically(directory / "config.json", [&](const fs::path& path) {
            std::ofstream output(path, std::ios::binary);
            output << Json(config).dump() << '\n';
            output.close();
            if (!output) throw std::runtime_error("Cannot write Qwen2 configuration");
        });
    } catch (...) {
        std::error_code ignored;
        for (const auto* name : {"model.safetensors", "tokenizer.json", "config.json"}) fs::remove(directory / name, ignored);
        fs::remove(directory, ignored);
        throw;
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "elman_import_weights INPUT OUTPUT.safetensors\n"
                         "elman_import_weights PYTORCH_MODEL_DIRECTORY OUTPUT_DIRECTORY\n"
                         "elman_import_weights INPUT.gguf MODEL_DIRECTORY --qwen2 [--tokenizer tokenizer.json]\n";
            return 0;
        }
        if (argc != 3 && argc != 4 && argc != 6)
            throw std::invalid_argument("Use --help for supported weight import arguments");
        const auto input = fs::u8path(argv[1]), output = fs::u8path(argv[2]);
        if (argc > 3) {
            if (model_file_extension(input) != ".gguf" || std::string(argv[3]) != "--qwen2" ||
                (argc == 6 && std::string(argv[4]) != "--tokenizer"))
                throw std::invalid_argument("Invalid Qwen2 conversion arguments");
            save_qwen2(GgufFile(input), output, argc == 6 ? fs::u8path(argv[5]) : fs::path{});
        } else if (fs::is_directory(input)) {
            save_pytorch_directory(input, output);
        } else {
            if (model_file_extension(output) != ".safetensors")
                throw std::invalid_argument("Weight conversion requires a .safetensors output");
            if (model_file_extension(input) == ".gguf") {
                const GgufFile file(input);
                Names names;
                for (const auto& entry : file.tensors()) names.emplace(entry.first, entry.first);
                save_gguf(file, names, output);
            } else save_safetensors(output, load_tensor_archive(input, Backend::CPU));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "elman_import_weights: " << error.what() << '\n';
        return 1;
    }
}
