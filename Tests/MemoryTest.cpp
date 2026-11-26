#include "Core/Memory.h"
#include "Core/Tensor.h"
#include "Backends/CpuMath.h"
#include <atomic>
#include <functional>
#include <iostream>
#include <limits>
#include <thread>

namespace {
using namespace elman;
size_t checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class Exception> void rejects(const std::function<void()>& run) {
    bool rejected = false;
    try { run(); } catch (const Exception&) { rejected = true; }
    check(rejected, "Expected memory rejection");
}
void accounting(Backend backend) {
    size_t before = memory_stats(backend).live_bytes();
    size_t bytes = memory_allocation_bytes(backend, 12);
    {
        MemoryLimit limit(backend, before + bytes);
        Tensor tensor = Tensor::zeros({3}, backend);
        check(memory_stats(backend).live_bytes() == before + bytes, "Tensor storage charge differs");
        Tensor alias = reshape(tensor, {1, 3});
        tensor = {};
        check(memory_stats(backend).live_bytes() == before + bytes, "Alias released live storage");
        uint64_t allocations = memory_stats(backend).allocations;
        rejects<MemoryLimitExceeded>([&] { Tensor::zeros({1}, backend); });
        check(memory_stats(backend).allocations == allocations, "Rejected allocation changed accounting");
        rejects<MemoryLimitExceeded>([&] { MemoryLimit impossible(backend, before); });
        alias = {};
        check(memory_stats(backend).live_bytes() == before, "Last alias did not release storage");
        check(limit.peak_bytes() == before + bytes, "Scoped peak was lost after release");
        MemoryLimit nested(backend, before);
        rejects<MemoryLimitExceeded>([&] { MemoryReservation reservation(backend, 1); });
    }
    {
        MemoryReservation first(backend, 64), second(backend, 32);
        second = std::move(first);
        check(memory_stats(backend).live_bytes() == before + 64, "Moving reservation leaked bytes");
        MemoryReservation third(std::move(second));
        check(memory_stats(backend).live_bytes() == before + 64, "Move construction changed live bytes");
    }
    check(memory_stats(backend).live_bytes() == before, "Reservation scope leaked");
    {
        BufferPtr empty = backend_allocate(backend, 0, DType::Float32);
        check(memory_stats(backend).live_bytes() == before + (backend == Backend::CPU ? 0 : 1), "Empty allocation charge differs");
    }
    // Allocation failure after charging must roll back the reservation.
    rejects<std::exception>([&] { backend_allocate(backend, std::numeric_limits<size_t>::max() / 2, DType::Float32); });
    check(memory_stats(backend).live_bytes() == before, "Failed allocation leaked charge");
    // Valid byte arithmetic, but an impossible physical allocation: exercises
    // rollback after the reservation has succeeded, rather than overflow alone.
    rejects<std::exception>([&] { backend_allocate(backend, std::numeric_limits<size_t>::max() - 15, DType::Bool); });
    check(memory_stats(backend).live_bytes() == before, "Physical allocation failure leaked charge");
    for (elman::DType dtype : all_dtypes()) for (size_t count : {size_t(0), size_t(1), size_t(3), size_t(17)}) {
        BufferPtr allocation = backend_allocate(backend, count, dtype);
        size_t payload = (size_t(dtype_info(dtype).bits) * count + 7) / 8;
        uint64_t charged = backend == Backend::CPU ? (payload + 7) / 8 * 8 : std::max(payload, size_t(1));
        check(allocation->bytes == payload && memory_stats(backend).live_bytes() == before + charged,
              "Packed/narrow dtype allocation charge differs");
    }
    check(memory_stats(backend).live_bytes() == before, "Dtype allocations leaked");
    std::atomic<bool> failed{false};
    {
        MemoryLimit limit(backend, before + 512);
        std::vector<std::thread> threads;
        for (int i = 0; i < 8; ++i) threads.emplace_back([&] {
            try {
                for (int j = 0; j < 1000; ++j) {
                    MemoryReservation reservation(backend, 64);
                    std::this_thread::yield();
                }
            } catch (...) { failed = true; }
        });
        for (std::thread& thread : threads) thread.join();
        check(!failed && limit.peak_bytes() <= before + 512, "Concurrent accounting exceeded its ceiling");
    }
    check(memory_stats(backend).live_bytes() == before, "Concurrent reservations leaked");
    MemoryInfo info = memory_info(backend);
    check(info.total_bytes > 0 && info.available_bytes <= info.total_bytes, "Invalid memory snapshot");
    check(memory_available_limit(backend, 1, 0) <= 1, "Explicit cap was not applied");
    check(memory_available_limit(backend, 0, std::numeric_limits<size_t>::max()) == before, "Reserve did not exhaust headroom");
}
void workspace(Backend backend) {
    Tensor input = Tensor::from_data({4097}, std::vector<float>(4097, 1), backend);
    size_t before = memory_stats(backend).live_bytes();
    {
        // MPS owns its internal scratch; exercise the tracked output limit there.
        // CPU/CUDA also reserve explicit sort workspace after allocating IDs.
        const size_t output_bytes = memory_allocation_bytes(backend, input.numel() * 4);
        MemoryLimit limit(backend, before + output_bytes - (backend == Backend::Metal ? 1 : 0));
        rejects<MemoryLimitExceeded>([&] { argsort(input, 0); });
    }
    check(memory_stats(backend).live_bytes() == before, "Sort workspace rejection leaked");
    check(argsort(input, 0).int_data().back() == 4096, "Sort did not recover");
    if (backend == Backend::CPU && cpu_vector_math_available()) {
        Tensor narrow = Tensor::ones({4097}, backend, false, DType::BFloat16);
        size_t baseline = memory_stats(backend).live_bytes();
        {
            MemoryLimit limit(backend, baseline + memory_allocation_bytes(backend, narrow.nbytes()));
            rejects<MemoryLimitExceeded>([&] { softmax(narrow, 0); });
        }
        check(memory_stats(backend).live_bytes() == baseline, "Narrow softmax workspace rejection leaked");
        check(softmax(narrow, 0).data().front() > 0, "Narrow softmax did not recover");
    }
    if (backend == Backend::CUDA) {
        {
            MemoryLimit limit(backend, before + input.nbytes());
            rejects<MemoryLimitExceeded>([&] { cumsum(input, 0); });
        }
        check(memory_stats(backend).live_bytes() == before, "Scan workspace rejection leaked");
        check(cumsum(input, 0).data().back() == 4097, "Scan did not recover");
        {
            Tensor matrix = Tensor::ones({64, 64}, backend);
            size_t baseline = memory_stats(backend).live_bytes();
            {
                MemoryLimit limit(backend, baseline + matrix.nbytes());
#ifdef ELMAN_TEST_CUBLAS
                rejects<MemoryLimitExceeded>([&] { matmul(matrix, matrix); });
#else
                check(matmul(matrix, matrix).data().front() == 64, "Workspace-free GEMM exceeded its output budget");
#endif
            }
            check(memory_stats(backend).live_bytes() == baseline, "Failed GEMM initialization leaked workspace");
            check(matmul(matrix, matrix).data().front() == 64, "GEMM did not recover");
#ifdef ELMAN_TEST_CUBLAS
            check(memory_stats(backend).live_bytes() == baseline + 1024 * 1024, "Persistent GEMM workspace was not charged");
#else
            check(memory_stats(backend).live_bytes() == baseline, "Native GEMM retained unexpected workspace");
#endif
        }
        {
            Tensor logits = Tensor::zeros({2, 3}, backend);
            Tensor targets = Tensor::from_int_data({2}, {0, 1}, backend);
            size_t baseline = memory_stats(backend).live_bytes();
            {
                MemoryLimit limit(backend, baseline + 4);
                rejects<MemoryLimitExceeded>([&] { cross_entropy(logits, targets); });
            }
            check(memory_stats(backend).live_bytes() == baseline, "Cross-entropy workspace rejection leaked");
            float loss = cross_entropy(logits, targets).data().front();
            check(loss > 1 && loss < 1.2f, "Cross-entropy did not recover");
        }
    }
}
void size_overflow() {
    rejects<std::overflow_error>([&] { checked_memory_add(std::numeric_limits<size_t>::max(), 1); });
    rejects<std::overflow_error>([&] { checked_memory_mul(std::numeric_limits<size_t>::max(), 2); });
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_memory_tests cpu|cuda|metal");
        Backend backend = elman::parse_backend(argv[1]);
        if (!elman::backend_available(backend)) return 77;
        accounting(backend);
        workspace(backend);
        size_overflow();
        std::cout << "Memory: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
