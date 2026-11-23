#include "Core/Tensor.h"
#include "Core/Execution.h"
#include "IO/Json.h"

#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
using namespace elman;
size_t checks = 0;
void check(bool condition, const std::string& message) {
    ++checks;
    if (!condition) throw std::runtime_error(message);
}
void rejects(const std::function<void()>& operation) {
    bool rejected = false;
    try { operation(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "invalid storage operation was accepted");
}
std::vector<uint8_t> unhex(const std::string& text) {
    check(text.size() % 2 == 0, "odd fixture hex size");
    std::vector<uint8_t> result;
    for (size_t i = 0; i < text.size(); i += 2) {
        size_t used = 0;
        uint64_t value = std::stoul(text.substr(i, 2), &used, 16);
        check(used == 2, "invalid fixture hex");
        result.push_back(uint8_t(value));
    }
    return result;
}
void roundtrip(Backend backend, DType dtype, Shape shape, const std::vector<uint8_t>& bytes) {
    Tensor source = Tensor::from_raw(shape, dtype, bytes);
    Tensor device = source.to(backend);
    check(device.backend() == backend && device.dtype() == dtype, "transfer changed tensor metadata");
    check(device.nbytes() == bytes.size(), "tensor byte size mismatch");
    check(device.raw_data() == bytes, "raw device round trip changed bytes");
    Tensor restored = device.to(Backend::CPU);
    check(restored.raw_data() == bytes && restored.shape() == shape, "device transfer lost precision or shape");
    check(reshape(device, {int(device.numel())}).raw_data() == bytes, "reshape changed storage bytes");
    std::vector<uint8_t> changed = bytes;
    for (unsigned char& byte : changed) byte ^= 0xa5;
    device.assign_raw(changed);
    check(device.raw_data() == changed, "raw assignment failed");
    rejects([&] { device.assign_raw(std::vector<uint8_t>(bytes.size() + 1)); });
}
void unit_tests(Backend backend) {
    std::mt19937 random(731);
    for (DType dtype : all_dtypes()) {
        check(parse_dtype(dtype_name(dtype)) == dtype, "dtype name does not round trip");
        for (int count : {0, 1, 3, 17, 257}) {
            size_t expected = (size_t(count) * dtype_info(dtype).bits + 7) / 8;
            check(dtype_storage_bytes(dtype, count) == expected, "dtype footprint is incorrect");
            BufferPtr zeros = backend_allocate(backend, count, dtype);
            check(backend_download_raw(zeros) == std::vector<uint8_t>(expected), "storage is not initialized");
            std::vector<uint8_t> bytes(expected);
            for (unsigned char& byte : bytes) byte = uint8_t(random());
            roundtrip(backend, dtype, {count}, bytes);
        }
    }
    uint64_t patterns[] = {0, 1, uint64_t(-1), 0x7fffffffffffffffULL, 0x8000000000000000ULL,
                           0x0020000000000001ULL, 0x7ff8000000000731ULL, 0x8000000000000001ULL};
    std::vector<uint8_t> exact(sizeof(patterns));
    std::memcpy(exact.data(), patterns, sizeof(patterns));
    for (elman::DType dtype : {DType::Int64, DType::UInt64, DType::Float64}) roundtrip(backend, dtype, {8}, exact);
    roundtrip(backend, DType::Complex128, {4}, exact);
    roundtrip(backend, DType::Float64, {}, std::vector<uint8_t>(8, 0x81));
    Tensor boolean = Tensor::from_int_data({5}, {0, 1, -1, 256, 0}, backend, DType::Bool);
    check(boolean.nbytes() == 5 && boolean.raw_data() == std::vector<uint8_t>({0, 1, 1, 1, 0}), "bool is not stored as bytes");
    check(logical_not(boolean).raw_data() == std::vector<uint8_t>({1, 0, 0, 0, 1}), "bool kernel storage mismatch");
    check(where(boolean, boolean, logical_not(boolean)).raw_data() == std::vector<uint8_t>({1, 1, 1, 1, 1}), "bool where copy mismatch");
    check(dtype_storage_bytes(DType::UInt1, size_t(-1)) == size_t(-1) / 8 + 1, "packed byte count overflow");
    rejects([&] { dtype_storage_bytes(DType::Complex128, size_t(-1)); });
    rejects([&] { dtype_info(static_cast<DType>(-1)); });
    rejects([&] { parse_dtype("invalid"); });
    rejects([&] { Tensor::from_raw({3}, DType::Float64, {1, 2}, backend); });
    Tensor half = Tensor::from_raw({1}, DType::Float16, {0, 0x3c}, backend);
    check(add(half, half).data() == std::vector<float>{2}, "half arithmetic integration");
    rejects([&] { half.int_data(); });
    check(half.data() == std::vector<float>{1}, "half host conversion");
    Tensor x = Tensor::from_data({2}, {1, 2}, backend, true);
    Tensor y = sum_all(mul(x, x));
    x.assign_raw(x.raw_data());
    rejects([&] { y.backward(); });
}
}

int main(int argc, char** argv) {
    try {
        Backend backend = elman::parse_backend(argc > 1 ? argv[1] : "cpu");
        if (!elman::backend_available(backend)) return 77;
        bool staging = argc > 3 && std::string(argv[3]) == "--pinned-staging";
        std::unique_ptr<elman::ExecutionStream> stream;
        std::unique_ptr<elman::StreamScope> execution;
        if (staging) {
            stream = std::make_unique<elman::ExecutionStream>(backend);
            execution = std::make_unique<elman::StreamScope>(*stream, true, true);
        }
        unit_tests(backend);
        size_t reference_cases = 0;
        if (argc > 2) {
            Json fixture = elman::Json::read(argv[2]);
            for (const elman::Json& item : fixture.at("cases").array()) {
                elman::Shape shape;
                for (const elman::Json& dim : item.at("shape").array()) shape.push_back(dim.integer());
                roundtrip(backend, elman::parse_dtype(item.at("dtype").string()), shape, unhex(item.at("hex").string()));
                ++reference_cases;
            }
        }
        if (stream) stream->synchronize();
        if (staging && backend == elman::Backend::CUDA) {
            HostStagingStats stats = stream->host_staging_stats();
            check(stats.uploads > 0 && stats.downloads > 0 && stats.reuses > 0, "Independent fixtures did not exercise pinned staging");
        }
        std::cout << "Storage checks passed: " << checks << "; independent fixtures: " << reference_cases << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
