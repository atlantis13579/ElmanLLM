#include "IO/checkpoint.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>

namespace {
int checks = 0;
void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& operation, const std::string& message) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    check(rejected, message);
}
void near(const std::vector<float>& actual, const std::vector<float>& expected, float tolerance, const std::string& message) {
    check(actual.size() == expected.size(), message + " size");
    for (size_t i = 0; i < actual.size(); ++i) check(std::isfinite(actual[i]) && std::abs(actual[i] - expected[i]) <= tolerance, message + " value " + std::to_string(i));
}

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        path = std::filesystem::temp_directory_path() / ("elman_model_io_" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        if (!std::filesystem::create_directory(path)) throw std::runtime_error("Cannot create unique model test directory");
    }
    ~TemporaryDirectory() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
};

elman::Json gpt_definition() {
    return elman::Json::parse(R"({"format":"elman.model","version":1,"name":"TestGPT2","architecture":"gpt2","vocab_size":8,"context_length":3,"embedding_dim":4,"num_heads":2,"num_layers":1,"mlp_ratio":2,"dropout":0,"layer_norm_epsilon":0.00001})");
}
elman::TrainingState state_for(const elman::Model& model) {
    elman::TrainingState result;
    result.optimizer = "adam";
    result.step = 7;
    for (const auto& parameter : model.parameters()) {
        result.first_moment.emplace_back(parameter.value.numel(), 0.125f);
        result.second_moment.emplace_back(parameter.value.numel(), 0.25f);
    }
    return result;
}
std::vector<char> read_bytes(const std::filesystem::path& path) {
    std::ifstream file(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
void write_bytes(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream file(path, std::ios::binary);
    file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!file) throw std::runtime_error("Cannot write checkpoint test fixture");
}

void json_tests() {
    using elman::Json;
    auto value = Json::parse(R"({"a":[1,-2.5e2,true,false,null,"line\nend"],"unicode":"\u4e2d\u6587\ud83d\ude00"})");
    check(value.at("a").at(0).integer() == 1, "JSON integer");
    check(value.at("a").at(1).number() == -250, "JSON exponent");
    check(value.at("a").at(2).boolean() && !value.at("a").at(3).boolean(), "JSON booleans");
    check(value.at("a").at(4).is_null(), "JSON null");
    check(value.at("a").at(5).string() == "line\nend", "JSON escape");
    check(value.at("unicode").string() == "\xe4\xb8\xad\xe6\x96\x87\xf0\x9f\x98\x80", "JSON surrogate-pair UTF8");
    check(Json::parse(value.dump()).dump(0) == value.dump(0), "JSON serialization roundtrip");
    for (const auto& invalid : {"", "[1,]", "{\"a\":1,}", "{\"a\":1,\"a\":2}", "01", "-01", ".5", "1.", "1e", "1e+", "NaN", "Infinity", "1e9999", "true false", "\"\\uD800\"", "\"\\uDC00\"", "\"\\uD800\\u1234\"", "\"\\q\"", "\"unterminated", "\"\n\""})
        rejects([&] { Json::parse(invalid); }, std::string("JSON must reject: ") + invalid);
    rejects([] { Json::parse("2147483648").integer(); }, "JSON integer range");
    rejects([] { Json::parse("1.25").integer(); }, "JSON fractional integer");
    rejects([] { Json::parse("false").number(); }, "JSON strict types");
    rejects([] { Json::parse(std::string(130, '[') + "0" + std::string(130, ']')); }, "JSON nesting bound");
}

void model_tests() {
    using namespace elman;
    auto definition = gpt_definition();
    auto model = create_model(definition, Backend::CPU, 7);
    auto input = Tensor::from_data({1, 3}, {1, 2, 3}, Backend::CPU, false, DType::Int32);
    auto output = model->forward(input, false);
    check(output.shape() == Shape({1, 3, 8}), "GPT2 output shape");
    check(!output.requires_grad(), "Inference does not record parameter gradients");
    auto altered = model->forward(Tensor::from_data({1, 3}, {1, 2, 5}, Backend::CPU, false, DType::Int32), false).data();
    auto original = output.data();
    near(std::vector<float>(altered.begin(), altered.begin() + 16), std::vector<float>(original.begin(), original.begin() + 16), 1e-6f, "Causal attention cannot observe future tokens");
    float final_change = 0;
    for (size_t i = 16; i < original.size(); ++i) final_change = std::max(final_change, std::abs(original[i] - altered[i]));
    check(final_change > 1e-5f, "Changed current token affects its own logits");
    check(model->parameters().size() == 16, "GPT2 tied embeddings are registered once");
    auto bad_heads = definition.object(); bad_heads["num_heads"] = Json(3);
    rejects([&] { create_model(Json(bad_heads), Backend::CPU); }, "GPT2 head divisibility validation");
    rejects([&] { model->forward(Tensor::zeros({1, 4}), false); }, "GPT2 context overflow validation");

    auto alex_definition = Json::parse(R"({"format":"elman.model","version":1,"name":"TestAlexNet","architecture":"alexnet","input_shape":[3,8,8],"channels":[2,2,2,2,2],"kernels":[3,3,3,3,3],"strides":[1,1,1,1,1],"paddings":[1,1,1,1,1],"pool_kernel":2,"pool_stride":2,"hidden_features":[4,4],"num_classes":3,"dropout":0})");
    auto alex = create_model(alex_definition, Backend::CPU, 9);
    auto classification = alex->forward(Tensor::ones({2, 3, 8, 8}), false);
    check(classification.shape() == Shape({2, 3}), "AlexNet output shape");
    check(alex->parameters().size() == 16, "AlexNet five convolution and three linear parameters");
    for (float value : classification.data()) check(std::isfinite(value), "AlexNet finite logits");
}

void checkpoint_tests() {
    using namespace elman;
    TemporaryDirectory temporary;
    const auto good_path = temporary.path / "valid.elman";
    const auto bad_path = temporary.path / "invalid.elman";
    auto definition = gpt_definition();
    auto model = create_model(definition, Backend::CPU, 7);
    auto state = state_for(*model);
    save_checkpoint(good_path.string(), *model, definition, state);
    auto restored = create_model(definition, Backend::CPU, 70);
    TrainingState loaded;
    load_checkpoint(good_path.string(), *restored, definition, &loaded);
    check(loaded.step == 7 && loaded.optimizer == "adam", "Checkpoint optimizer metadata");
    for (size_t i = 0; i < model->parameters().size(); ++i) {
        near(restored->parameters()[i].value.data(), model->parameters()[i].value.data(), 0, "Checkpoint exact parameter restore");
        near(loaded.first_moment[i], state.first_moment[i], 0, "Checkpoint first moments");
        near(loaded.second_moment[i], state.second_moment[i], 0, "Checkpoint second moments");
    }
    auto input = Tensor::from_data({1, 3}, {1, 2, 3}, Backend::CPU, false, DType::Int32);
    near(restored->forward(input, false).data(), model->forward(input, false).data(), 0, "Checkpoint inference equivalence");
    save_checkpoint(good_path.string(), *model, definition, state);
    check(!std::filesystem::exists(good_path.string() + ".tmp"), "Checkpoint temporary file removed after replacement");
    auto bytes = read_bytes(good_path);
    check(bytes.size() > 100, "Checkpoint nonempty");
    auto corrupted = bytes; corrupted[0] ^= 1; write_bytes(bad_path, corrupted);
    rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); }, "Checkpoint signature validation");
    corrupted = bytes; corrupted[8] = 2; write_bytes(bad_path, corrupted);
    rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); }, "Checkpoint version validation");
    corrupted = bytes; corrupted[corrupted.size() - 1] ^= 1; write_bytes(bad_path, corrupted);
    auto previous_values = restored->parameters()[0].value.data();
    for (float& value : previous_values) value += 1;
    restored->parameters()[0].value.assign(previous_values);
    rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); }, "Checkpoint checksum validation");
    near(restored->parameters()[0].value.data(), previous_values, 0, "Invalid checkpoint does not mutate parameters");
    corrupted = bytes; corrupted.resize(corrupted.size() / 2); write_bytes(bad_path, corrupted);
    rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); }, "Truncated checkpoint validation");
    corrupted = bytes; corrupted.push_back(0); write_bytes(bad_path, corrupted);
    rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); }, "Trailing checkpoint bytes validation");
    auto changed_definition = definition.object(); changed_definition["layer_norm_epsilon"] = Json(0.001);
    rejects([&] { load_checkpoint(good_path.string(), *restored, Json(changed_definition)); }, "Checkpoint model configuration validation");
    changed_definition = definition.object(); changed_definition["name"] = Json("OtherModel");
    auto other = create_model(Json(changed_definition), Backend::CPU, 7);
    rejects([&] { load_checkpoint(good_path.string(), *other, Json(changed_definition)); }, "Checkpoint model name validation");
}

}

int main() {
    try {
        json_tests();
        model_tests();
        checkpoint_tests();
        std::cout << "Model/IO tests passed: " << checks << " checks\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Model/IO test failure after " << checks << " checks: " << error.what() << '\n';
        return 1;
    }
}
