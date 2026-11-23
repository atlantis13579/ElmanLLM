#include "Core/Execution.h"
#include "Core/AllocationPool.h"
#include "Core/Graph.h"
#include "Inference/LLM/Sampling.h"
#include <atomic>
#include <iostream>
#include <limits>
#include <thread>

#ifdef ELMAN_HAS_CUDA
int cuda_execution_ordering_checks();
#endif
namespace {
using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
void events(Backend backend) {
    ExecutionStream first(backend), second(backend);
    ExecutionEvent initial = first.record();
    initial.synchronize();
    check(initial.ready(), "Completed event remained pending");
    second.wait(initial);
    second.synchronize();
    check(second.ready(), "Empty stream remained pending");
    for (int action = 0; action < 3; ++action) {
        bool rejected = false;
        try {
            ExecutionEvent missing;
            if (action == 0) missing.ready();
            else if (action == 1) missing.synchronize();
            else first.wait(missing);
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Unrecorded event accepted");
    }
    if (backend == Backend::CUDA) {
        bool rejected = false;
        try { first.wait(ExecutionStream(Backend::CPU).record()); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Cross-backend event accepted");
    }
    check(!asynchronous_execution(backend), "Unexpected initial execution binding");
    {
        StreamScope outer(first);
        check(asynchronous_execution(backend) == (backend == Backend::CUDA), "Asynchronous binding missing");
        try {
            StreamScope inner(second, false);
            check(!asynchronous_execution(backend), "Synchronous override ignored");
            throw std::runtime_error("unwind stream");
        } catch (const std::runtime_error& error) {
            if (std::string(error.what()) != "unwind stream") throw;
        }
        check(asynchronous_execution(backend) == (backend == Backend::CUDA), "Exception did not restore parent binding");
    }
    check(!asynchronous_execution(backend), "Scope did not restore default binding");
}
void numerical(Backend backend) {
    size_t before = memory_stats(backend).live_bytes();
    for (bool reuse : {false, true}) for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        ExecutionStream first(backend), second(backend);
        AllocationPool pool(backend, 0, reuse);
        for (int repeat = 0; repeat < 4; ++repeat) {
            Tensor x = Tensor::from_data({4}, {1, -2, 3, -4}, backend, true, dtype);
            Tensor square, result;
            {
                StreamScope scope(first);
                square = mul(x, x);
            }
            {
                StreamScope scope(second);
                result = add(square, square);
            }
            {
                StreamScope scope(first);
                sum(result).backward();
            }
            check(result.data() == std::vector<float>({2, 8, 18, 32}), "Cross-stream forward differs");
            check(x.grad() == std::vector<float>({4, -8, 12, -16}), "Cross-stream backward differs");
            ExecutionEvent old = first.record();
            x.assign({5, 6, 7, 8});
            old.synchronize();
            check(x.data() == std::vector<float>({5, 6, 7, 8}), "Host assignment differs");
        }
        first.synchronize(); second.synchronize();
        check((pool.stats().reuses > 0) == reuse, "Stream scope bypassed pool selection");
    }
    check(memory_stats(backend).live_bytes() == before, "Stream arithmetic leaked storage");
}
void escaped(Backend backend) {
    size_t before = memory_stats(backend).live_bytes();
    Tensor result;
    {
        ExecutionStream stream(backend);
        StreamScope scope(stream);
        AllocationPool pool(backend);
        Tensor input = Tensor::from_data({4}, {1, 2, 3, 4}, backend);
        result = mul(input, input);
    }
    check(result.data() == std::vector<float>({1, 4, 9, 16}), "Result did not survive stream destruction");
    std::thread release([value = std::move(result)]() mutable { value = {}; });
    release.join();
    check(memory_stats(backend).live_bytes() == before, "Escaped result leaked storage");
}
void matrices(Backend backend) {
    size_t before = memory_stats(backend).live_bytes();
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        std::vector<float> input(8 * 16), weights(12 * 16), bias(12), expected(8 * 12);
        for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i % 5) - 2);
        for (size_t i = 0; i < weights.size(); ++i) weights[i] = float(int(i % 3) - 1);
        for (int i = 0; i < 12; ++i) bias[i] = float(i - 6);
        for (int row = 0; row < 8; ++row) for (int column = 0; column < 12; ++column) {
            float& value = expected[row * 12 + column];
            value = bias[column];
            for (int k = 0; k < 16; ++k) value += input[row * 16 + k] * weights[column * 16 + k];
        }
        ExecutionStream first(backend), second(backend);
        AllocationPool pool(backend);
        Tensor a = Tensor::from_data({8, 16}, input, backend, false, dtype);
        Tensor w = Tensor::from_data({12, 16}, weights, backend, false, dtype);
        Tensor b = Tensor::from_data({12}, bias, backend, false, dtype);
        for (float& value : bias) value += 2;
        Tensor other_bias = Tensor::from_data({12}, bias, backend, false, dtype);
        std::vector<float> other_expected = expected;
        for (float& value : other_expected) value += 2;
        for (int repeat = 0; repeat < 6; ++repeat) {
            Tensor x, y;
            { StreamScope scope(first); x = linear(a, w, repeat % 2 ? other_bias : b); }
            { StreamScope scope(second); y = linear(a, w, repeat % 2 ? b : other_bias); }
            check(x.data() == (repeat % 2 ? other_expected : expected) &&
                  y.data() == (repeat % 2 ? expected : other_expected), "Per-stream biased GEMM contexts interfered");
        }
    }
    check(memory_stats(backend).live_bytes() == before, "Stream vendor contexts leaked accounted workspaces");
}
void threads(Backend backend) {
    size_t before = memory_stats(backend).live_bytes();
    {
        Tensor input = Tensor::from_data({4}, {1, 2, 3, 4}, backend);
        std::atomic<int> failures{0};
        std::vector<std::thread> workers;
        for (int i = 0; i < 4; ++i) workers.emplace_back([&] {
            try {
                ExecutionStream stream(backend);
                StreamScope scope(stream);
                AllocationPool pool(backend);
                for (int repeat = 0; repeat < 12; ++repeat)
                    if (mul(input, input).data() != std::vector<float>({1, 4, 9, 16})) ++failures;
                stream.synchronize();
            } catch (...) { ++failures; }
        });
        for (std::thread& worker : workers) worker.join();
        check(failures == 0, "Concurrent readers interfered");
    }
    check(memory_stats(backend).live_bytes() == before, "Concurrent streams leaked storage");
}
void greedy_selection(Backend backend) {
    ExecutionStream producer(backend), consumer(backend);
    AllocationPool pool(backend);
    std::vector<float> values(151938, -50.0f);
    values.front() = values.back() = 100.0f;
    values[256] = values[65792] = 20.0f;
    Tensor row;
    {
        StreamScope scope(producer);
        Tensor storage = Tensor::from_data({int(values.size())}, values, backend);
        row = contiguous_view(storage, {151936}, 1);
    }
    StreamScope scope(consumer);
    check(greedy_token(row) == 255, "Greedy selection lost the first tied ID or the view offset");
    check(greedy_token(Tensor::from_data({2}, {-0.0f, 0.0f}, backend)) == 0, "Signed zero tie changed ID");
    check(greedy_token(Tensor::from_data({1}, {-2.0f}, backend)) == 0, "Single-token vocabulary failed");
    for (float invalid : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        values[130000] = invalid;
        bool rejected = false;
        try { greedy_token(Tensor::from_data({int(values.size())}, values, backend)); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Greedy selection accepted a nonfinite logit");
    }
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_execution_tests cpu|cuda");
        Backend backend = elman::parse_backend(argv[1]);
        if (!elman::backend_available(backend)) return 77;
        events(backend); numerical(backend); matrices(backend); escaped(backend); threads(backend); greedy_selection(backend);
#ifdef ELMAN_HAS_CUDA
        if (backend == elman::Backend::CUDA) checks += cuda_execution_ordering_checks();
#endif
        std::cout << "Execution streams: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
