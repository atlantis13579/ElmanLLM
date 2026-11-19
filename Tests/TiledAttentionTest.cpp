#include "Core/Tensor.h"
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
void bounded(Backend backend, DType dtype, int queries, bool reuse, bool asynchronous, AttentionTiling tiling) {
    auto before = memory_stats(backend).live_bytes();
    {
        ExecutionStream stream(backend);
        StreamScope scope(stream, asynchronous);
        AllocationPool pool(backend, 0, reuse);
        // Initialize the selected stream's vendor workspace before admission.
        matmul(Tensor::ones({2, 2}, backend), Tensor::ones({2, 2}, backend)).data();
        auto q = Tensor::zeros({1, 4, queries, 8}, backend, false, dtype);
        auto k = Tensor::zeros({1, 2, 512, 8}, backend, false, dtype);
        auto v = Tensor::from_data({1, 2, 512, 8}, std::vector<float>(2 * 512 * 8, 2), backend, false, dtype);
        auto mask = Tensor::zeros({}, backend, false, dtype);
        pool.clear();
        auto base = memory_stats(backend).live_bytes();
        size_t width = dtype_storage_bytes(dtype, 1);
        size_t output = memory_allocation_bytes(backend, size_t(4) * queries * 8 * width);
        size_t scratch = memory_allocation_bytes(backend, size_t(tiling == AttentionTiling::Heads ? queries : 4 * 32) * 512 * width);
        // Row math, pointer arrays and allocator alignment are charged too.
        size_t budget = base + output + scratch + 16 * 1024;
        {
            MemoryLimit limit(backend, budget);
            auto dispatches = backend_dispatch_count(backend);
            auto result = tiled_attention(q, k, v, mask, 2, 32, false, 0, tiling);
            check(backend_dispatch_count(backend) == dispatches + 1, "Tiled attention did not use one native dispatch");
            auto values = result.data();
            check(std::all_of(values.begin(), values.end(), [](float x) { return x == 2; }), "Bounded attention output differs");
            check(limit.peak_bytes() <= budget, "Tiled attention exceeded its selected workspace bound");
            result = {};
            bool rejected = false;
            try {
                auto repeated = reshape(expand(reshape(k, {1, 2, 1, 512, 8}), {1, 2, 2, 512, 8}), {1, 4, 512, 8});
                matmul_transposed(q, repeated).data();
            } catch (const MemoryLimitExceeded&) { rejected = true; }
            check(rejected, "Dense scores unexpectedly fit the tiled workspace budget");
            check(tiled_attention(q, k, v, mask, 2, 32, false, 0, tiling).data() == values, "Attention failed to recover after dense rejection");
        }
        stream.synchronize();
    }
    check(memory_stats(backend).live_bytes() == before, "Tiled attention leaked stream or workspace storage");
}
void fourth_input(Backend backend, bool reuse) {
    auto before = memory_stats(backend).live_bytes();
    {
        ExecutionStream producer(backend), consumer(backend);
        AllocationPool pool(backend, 0, reuse);
        auto q = Tensor::zeros({1, 4, 33, 8}, backend);
        auto k = Tensor::zeros({1, 2, 2, 8}, backend);
        auto v = Tensor::from_data({1, 2, 2, 1}, {2, 6, 3, 7}, backend);
        std::vector<float> initial(66), replacement(66), expected(4 * 33);
        for (int i = 0; i < 33; ++i) {
            initial[2 * i + 1] = -std::numeric_limits<float>::infinity();
            replacement[2 * i] = -std::numeric_limits<float>::infinity();
        }
        for (int head = 0; head < 4; ++head)
            for (int i = 0; i < 33; ++i) expected[head * 33 + i] = head < 2 ? 2 : 3;
        auto source = Tensor::from_data({1, 1, 33, 2}, initial, backend);
        for (int repeat = 0; repeat < 4; ++repeat) {
            Tensor mask, result;
            { StreamScope scope(producer); mask = add(source, 0); }
            { StreamScope scope(consumer); result = tiled_attention(q, k, v, mask, 2, 8); }
            { StreamScope scope(producer); mask.assign(replacement); }
            mask = {};
            check(result.data() == expected, "Fourth-input stream ordering or lifetime differs");
        }
        producer.synchronize(); consumer.synchronize();
        StreamScope scope(producer);
        auto mask = Tensor::zeros({}, backend, true);
        auto result = tiled_attention(q, k, v, mask);
        mask.assign({1});
        bool rejected = false;
        try { result.backward(Tensor::ones(result.shape(), backend)); }
        catch (const std::exception& error) { rejected = std::string(error.what()).find("modified before backward") != std::string::npos; }
        check(rejected, "Fourth-input mutation escaped the autograd version check");
    }
    check(memory_stats(backend).live_bytes() == before, "Fourth-input test leaked storage");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_attention_tests cpu|cuda");
        auto backend = parse_backend(argv[1]);
        if (!backend_available(backend)) return 77;
        for (bool reuse : {false, true}) {
            fourth_input(backend, reuse);
            for (bool asynchronous : {false, true}) {
                if (backend == Backend::CPU && asynchronous) continue;
                for (auto dtype : {DType::Float32, DType::Float16, DType::BFloat16})
                    for (int queries : {257, 513})
                        for (auto tiling : {AttentionTiling::Queries, AttentionTiling::Heads})
                            bounded(backend, dtype, queries, reuse, asynchronous, tiling);
            }
        }
        std::cout << "Tiled attention memory/lifetime: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
