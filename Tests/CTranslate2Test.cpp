#include "IO/CTranslate2.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "Malformed container was accepted");
}
void integer(std::ostream& out, uint64_t value, int width) {
    for (int i = 0; i < width; ++i) out.put(char(value >> (8 * i)));
}
void name(std::ostream& out, const std::string& value) {
    integer(out, value.size() + 1, 2);
    out.write(value.c_str(), value.size() + 1);
}
void tensor(std::ostream& out, const std::string& key, int type, const Tensor& value) {
    name(out, key);
    integer(out, value.shape().size(), 1);
    for (int dimension : value.shape()) integer(out, dimension, 4);
    integer(out, type, 1);
    const auto bytes = value.raw_data();
    integer(out, bytes.size(), 4);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
}
void fixture(const std::filesystem::path& path, bool cycle = false) {
    std::ofstream out(path, std::ios::binary);
    integer(out, 6, 4);
    name(out, "WhisperSpec");
    integer(out, 3, 4);
    integer(out, 4, 4);
    tensor(out, "weight", 1, cast(Tensor::from_int_data({2, 3}, {2, -4, 6, 12, -16, 20}), DType::Int8));
    tensor(out, "weight_scale", 0, Tensor::from_data({2}, {2, 4}));
    tensor(out, "bias", 4, Tensor::from_data({2}, {.25f, -.5f}, Backend::CPU, false, DType::Float16));
    tensor(out, "heads", 2, cast(Tensor::from_int_data({}, {20}), DType::Int16));
    integer(out, 2, 4);
    name(out, "tied"); name(out, cycle ? "chained" : "bias");
    name(out, "chained"); name(out, "tied");
    check(bool(out), "Could not write fixture");
}
void near(const Tensor& value, const std::vector<float>& expected) {
    const auto actual = cast(value, DType::Float32).data();
    check(actual.size() == expected.size(), "Unexpected tensor size");
    for (size_t i = 0; i < actual.size(); ++i)
        check(std::abs(actual[i] - expected[i]) < 1e-5, "Unexpected tensor value");
}
}

int main(int argc, char** argv) {
    try {
        check(argc == 3, "Usage: elman_ctranslate2_tests BACKEND DIRECTORY");
        const Backend backend = parse_backend(argv[1]);
        if (!backend_available(backend)) return 77;
        const auto directory = std::filesystem::u8path(argv[2]);
        std::filesystem::create_directories(directory);
        const auto path = directory / "model.bin";
        fixture(path);
        const CTranslate2File file(path);
        check(file.specification() == "WhisperSpec" && file.revision() == 3, "Lost container metadata");
        check(file.tensors().size() == 6, "Lost aliases");
        check(file.info("chained").offset == file.info("bias").offset, "Alias payload was not shared");
        check(file.info("heads").shape.empty(), "Scalar shape was not preserved");
        near(file.load("heads", backend), {20});
        near(file.load_float("weight", backend, DType::Float32), {1, -2, 3, 3, -4, 5});
        near(file.load_float("chained", backend, DType::Float32), {.25f, -.5f});
        rejects([&] { file.load("absent", backend); });
        rejects([&] { file.load_float("weight", backend, DType::Int32); });
        rejects([&] { file.load_float("heads", backend, DType::Float32); });
        fixture(path, true);
        rejects([&] { CTranslate2File invalid(path); });
        fixture(path);
        {
            std::ofstream append(path, std::ios::binary | std::ios::app);
            append.put('!');
        }
        rejects([&] { CTranslate2File invalid(path); });
        fixture(path);
        std::filesystem::resize_file(path, std::filesystem::file_size(path) - 3);
        rejects([&] { CTranslate2File invalid(path); });
        fixture(path);
        {
            std::fstream change(path, std::ios::binary | std::ios::in | std::ios::out);
            integer(change, 7, 4);
        }
        rejects([&] { CTranslate2File invalid(path); });
        std::cout << "CTranslate2 container checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
