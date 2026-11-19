#include "Core/AllocationPool.h"
#include "Core/Graph.h"
#include "Core/Execution.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace elman;
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}
void storage(Backend backend, bool pooled) {
    auto baseline = memory_stats(backend).live_bytes();
    {
        AllocationPool pool(backend, 0, pooled);
        for (auto dtype : all_dtypes()) {
            std::vector<uint8_t> data(dtype_storage_bytes(dtype, 40));
            for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t(i * 73 + 19);
            auto source = Tensor::from_raw({40}, dtype, data, backend);
            size_t offset = dtype_storage_bytes(dtype, 8), length = dtype_storage_bytes(dtype, 16);
            const auto requests = pool.stats().requests;
            const auto live = memory_stats(backend).live_bytes();
            auto view = contiguous_view(source, {4, 4}, 8);
            auto nested = contiguous_view(view, {8}, 8);
            check(pool.stats().requests == requests && memory_stats(backend).live_bytes() == live,
                  "View allocated tensor storage");
            check(view.raw_data() == std::vector<uint8_t>(data.begin() + offset, data.begin() + offset + length), "View byte range differs");
            auto version = source.node()->value->version;
            std::vector<uint8_t> replacement(length, 0x5a);
            view.assign_raw(replacement);
            std::copy(replacement.begin(), replacement.end(), data.begin() + offset);
            check(source.raw_data() == data, "View assignment changed the wrong source range");
            check(source.node()->value->version == version + 1 && nested.node()->value->version == version + 1,
                  "View mutation version is not shared");
            auto expected = nested.raw_data();
            source = {}; view = {};
            auto other = Tensor::from_raw({40}, dtype, std::vector<uint8_t>(data.size(), 0), backend);
            check(nested.raw_data() == expected, "Pool reused storage with a surviving view");
            other = {}; nested = {};
            pool.clear();
        }
        for (auto dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
            auto x = Tensor::zeros({8}, backend, false, dtype);
            check(contiguous_view(x, {0}, 8).raw_data().empty(), "Empty terminal view differs");
            check(contiguous_view(x, {}, 3).data() == std::vector<float>{0}, "Scalar view differs");
            rejects([&] { contiguous_view(x, {1}, 8); }, "Out-of-range view accepted");
            rejects([&] { contiguous_view(x, {0}, 9); }, "Empty view beyond storage accepted");
            rejects([&] { contiguous_view(x, {1}, std::numeric_limits<size_t>::max()); }, "Overflowing offset accepted");
            rejects([&] { contiguous_view(x, {-1}); }, "Inferred view shape accepted");
        }
        rejects([] { contiguous_view(Tensor{}, {0}); }, "Undefined source accepted");
        for (auto dtype : all_dtypes()) if (dtype_info(dtype).bits % 8) {
            auto x = Tensor::from_raw({33}, dtype, std::vector<uint8_t>(dtype_storage_bytes(dtype, 33), 0x7b), backend);
            rejects([&] { contiguous_view(x, {8}, 1); }, "Unaligned packed offset accepted");
            rejects([&] { contiguous_view(x, {1}, 8); }, "Packed view can overwrite a neighboring logical value");
            auto tail = contiguous_view(x, {1}, 32);
            check(tail.raw_data() == std::vector<uint8_t>(dtype_storage_bytes(dtype, 1), 0x7b), "Packed terminal view differs");
        }
    }
    check(memory_stats(backend).live_bytes() == baseline, "View storage leaked");
}
void gradients(Backend backend) {
    for (auto dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        auto x = Tensor::ones({4, 5}, backend, true, dtype);
        auto view = contiguous_view(x, {3, 4}, 3);
        sum_all(mul(view, view)).backward();
        std::vector<float> expected(20);
        std::fill(expected.begin() + 3, expected.begin() + 15, 2.0f);
        check(x.grad() == expected, "View gradient did not scatter to the source range");
        for (int mutation = 0; mutation < 3; ++mutation) {
            auto source = Tensor::ones({16}, backend, true, dtype);
            auto selected = contiguous_view(source, {4}, 4);
            auto output = sum_all(mul(selected, selected));
            if (mutation == 0) source.assign(std::vector<float>(16, 2));
            else if (mutation == 1) selected.assign(std::vector<float>(4, 2));
            else contiguous_view(source, {4}, 12).assign(std::vector<float>(4, 2));
            bool rejected = false;
            try { output.backward(); }
            catch (const std::exception& error) { rejected = std::string(error.what()).find("modified before backward") != std::string::npos; }
            check(rejected, "Aliased mutation escaped autograd version checks");
        }
    }
}
void output_view(Backend backend) {
    auto root = Tensor::zeros({12}, backend), view = contiguous_view(root, {4}, 3);
    auto a = Tensor::ones({4}, backend), b = Tensor::ones({4}, backend);
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = 4;
    p.v[KernelParams::kOutputRankSlot] = 1;
    p.v[KernelParams::kOutputShapeSlot] = 4;
    for (auto input : {KernelInput::A, KernelInput::B, KernelInput::C, KernelInput::D})
        p.v[KernelParams::input_slot(input)] = -1;
    for (auto input : {KernelInput::A, KernelInput::B}) {
        int base = KernelParams::input_slot(input);
        p.v[base] = 1;
        p.v[base + KernelParams::kInputShapeOffset] = 4;
        p.v[base + KernelParams::kInputStrideOffset] = 1;
    }
    backend_dispatch("add", view.node()->value, {a.node()->value, b.node()->value}, p);
    std::vector<float> expected(12);
    std::fill(expected.begin() + 3, expected.begin() + 7, 2.0f);
    check(root.data() == expected, "Native output view did not preserve surrounding storage");
}
void streams(Backend backend, bool pooled) {
    auto before = memory_stats(backend).live_bytes();
    {
        ExecutionStream producer(backend), consumer(backend);
        AllocationPool pool(backend, 0, pooled);
        auto x = Tensor::ones({4096}, backend);
        for (int repeat = 0; repeat < 8; ++repeat) {
            Tensor root, view, result;
            { StreamScope scope(producer); root = mul(x, 3); view = contiguous_view(root, {2048}, 1024); }
            { StreamScope scope(consumer); result = mul(view, 5); }
            { StreamScope scope(producer); root.assign(std::vector<float>(4096, -7)); }
            root = {}; view = {};
            check(result.data() == std::vector<float>(2048, 15), "Root/view stream dependency was lost");
        }
        producer.synchronize(); consumer.synchronize();
    }
    check(memory_stats(backend).live_bytes() == before, "View stream lifetime leaked");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_view_tests cpu|cuda");
        auto backend = parse_backend(argv[1]);
        if (!backend_available(backend)) return 77;
        for (bool pooled : {false, true}) { storage(backend, pooled); streams(backend, pooled); }
        gradients(backend);
        output_view(backend);
        std::cout << "Contiguous views: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
