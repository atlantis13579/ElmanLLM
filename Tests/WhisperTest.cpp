#include "Models/Audio/Whisper.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
template<class F> void rejects(F action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    check(caught, "Invalid Whisper input was accepted");
}
void integer(std::ostream& out, uint64_t value, int bytes) {
    for (int i = 0; i < bytes; ++i) out.put(char(value >> (8 * i)));
}
void name(std::ostream& out, const std::string& value) {
    integer(out, value.size() + 1, 2); out.write(value.c_str(), value.size() + 1);
}
void fixture(const std::filesystem::path& directory) {
    std::vector<std::pair<std::string, Tensor>> tensors;
    auto add = [&](const std::string& key, const Shape& shape, bool norm = false) {
        size_t count = 1;
        for (int dim : shape) count *= dim;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) values[i] = norm ? 1.f : float(.1 * std::sin(i * .17 + tensors.size()));
        tensors.emplace_back(key, Tensor::from_data(shape, values));
    };
    auto norm = [&](const std::string& key) { add(key + "/gamma", {8}, true); add(key + "/beta", {8}); };
    auto linear = [&](const std::string& key, int input, int output) {
        add(key + "/weight", {output, input}); add(key + "/bias", {output});
    };
    add("encoder/conv1/weight", {8, 4, 3}); add("encoder/conv1/bias", {8});
    add("encoder/conv2/weight", {8, 8, 3}); add("encoder/conv2/bias", {8});
    add("encoder/position_encodings/encodings", {4, 8});
    add("decoder/position_encodings/encodings", {16, 8});
    add("decoder/embeddings/weight", {16, 8});
    for (const std::string stack : {"encoder", "decoder"}) {
        tensors.emplace_back(stack + "/num_heads", Tensor::from_data({}, {2}));
        norm(stack + "/layer_norm");
        for (int i = 0; i < 2; ++i) {
            const std::string key = stack + "/layer_" + std::to_string(i);
            norm(key + "/self_attention/layer_norm");
            linear(key + "/self_attention/linear_0", 8, 24);
            linear(key + "/self_attention/linear_1", 8, 8);
            norm(key + "/ffn/layer_norm");
            linear(key + "/ffn/linear_0", 8, 32);
            linear(key + "/ffn/linear_1", 32, 8);
            if (stack == "decoder") {
                norm(key + "/attention/layer_norm");
                linear(key + "/attention/linear_0", 8, 8);
                linear(key + "/attention/linear_1", 8, 16);
                linear(key + "/attention/linear_2", 8, 8);
            }
        }
    }
    std::ofstream file(directory / "model.bin", std::ios::binary);
    integer(file, 6, 4); name(file, "WhisperSpec"); integer(file, 3, 4); integer(file, tensors.size(), 4);
    for (const auto& entry : tensors) {
        name(file, entry.first); integer(file, entry.second.shape().size(), 1);
        for (int dim : entry.second.shape()) integer(file, dim, 4);
        integer(file, 0, 1);
        const auto bytes = entry.second.raw_data();
        integer(file, bytes.size(), 4); file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    }
    integer(file, 0, 4);
    check(bool(file), "Cannot write tiny Whisper fixture");
}
void near(const Tensor& a, const Tensor& b, float tolerance = 2e-5f) {
    check(a.shape() == b.shape(), "Whisper shape mismatch");
    const auto x = cast(a, DType::Float32).data(), y = cast(b, DType::Float32).data();
    for (size_t i = 0; i < x.size(); ++i)
        check(std::isfinite(x[i]) && std::abs(x[i] - y[i]) < tolerance, "Whisper numerical mismatch");
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "Usage: elman_whisper_tests BACKEND DIRECTORY");
        const auto backend = parse_backend(argv[1]);
        if (!backend_available(backend)) return 77;
        const auto directory = std::filesystem::u8path(argv[2]);
        std::filesystem::create_directories(directory); fixture(directory);
        const Whisper cpu(directory, Backend::CPU), model(directory, backend);
        std::vector<float> values(32);
        for (size_t i = 0; i < values.size(); ++i) values[i] = float(std::sin(i * .23));
        const auto mel = Tensor::from_data({4, 8}, values, backend);
        const auto encoded = model.encode(mel);
        const auto reference = cpu.encode(Tensor::from_data({4, 8}, values));
        near(encoded, reference);
        const auto memory = model.prepare_memory(encoded), cpu_memory = cpu.prepare_memory(reference);
        const auto complete = model.decode({1, 2, 3, 4}, memory, nullptr, {{0, 1}, {1, 0}});
        near(complete.logits, cpu.decode({1, 2, 3, 4}, cpu_memory).logits);
        check(complete.alignment.shape() == Shape{2, 4, 4}, "Alignment shape is incorrect");
        const auto prefix = model.decode({1, 2}, memory);
        const auto cached = model.decode({3, 4}, memory, &prefix.cache);
        near(complete.logits, cached.logits);
        const auto single = model.decode({3}, memory, &prefix.cache);
        near(model.decode({4}, memory, &single.cache).logits, complete.logits);
        near(model.decode({1, 2}, memory).logits, prefix.logits);
        check(prefix.cache.length == 2 && prefix.cache.layers[0].key.shape()[2] == 2,
              "Decoding mutated the shared prefix cache");
        near(slice(complete.cache.layers[0].key, 2, 0, 2), prefix.cache.layers[0].key);
        rejects([&] { model.decode({}, memory); });
        rejects([&] { model.decode({16}, memory); });
        rejects([&] { model.decode({1}, memory, nullptr, {{2, 0}}); });
        rejects([&] { model.decode(std::vector<int32_t>(17, 1), memory); });
        rejects([&] { model.encode(Tensor::zeros({4, 7}, backend)); });
        rejects([&] { model.encode(mel, [](int, int) { return false; }); });
        const Whisper half(directory, backend, DType::Float16);
        near(half.decode({1, 2, 3, 4}, half.prepare_memory(half.encode(mel))).logits, complete.logits, .003f);
        std::cout << "Whisper component checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
