#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Backends/CudaStream.h"
#include <atomic>
#include <chrono>
#include <future>
#include <thread>

namespace {
using namespace elman;
struct Gate {
    ExecutionStream& owner;
    explicit Gate(ExecutionStream& stream) : owner(stream) {}
    ~Gate() {
        try { owner.synchronize(); } catch (...) {}
    }
    static void CUDART_CB wait(void*) {
        // A finite delay never waits for unrelated CUDA work or calls CUDA.
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    void enqueue() { check_cuda_execution(cudaLaunchHostFunc(current_cuda_stream(), wait, nullptr), "Test host gate"); }
};
KernelParams addition() {
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = 16;
    p.v[KernelParams::kOutputRankSlot] = 1;
    p.v[KernelParams::kOutputShapeSlot] = 16;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32);
    for (int i = 0; i < 2; ++i) {
        int base = KernelParams::kInputRankSlot + i * KernelParams::kInputSlotWidth;
        p.v[base] = 1;
        p.v[base + KernelParams::kInputShapeOffset] = 16;
        p.v[base + KernelParams::kInputStrideOffset] = 1;
        p.v[KernelParams::kInputDTypeSlot + i] = int(DType::Float32);
    }
    return p;
}
BufferPtr values(float value) {
    auto buffer = backend_allocate(Backend::CUDA, 16, DType::Float32);
    backend_upload(buffer, std::vector<float>(16, value));
    return buffer;
}
}
int cuda_execution_ordering_checks() {
    int checks = 0;
    auto check = [&](bool valid, const char* message) {
        ++checks;
        if (!valid) throw std::runtime_error(message);
    };
    ExecutionStream first(Backend::CUDA), second(Backend::CUDA), third(Backend::CUDA);
    auto a = values(1), b = values(2), output = values(0), independent = values(0), dependent = values(0);
    auto params = addition();
    auto snapshot = first.record();
    {
        Gate gate(first);
        {
            StreamScope scope(first);
            gate.enqueue();
            backend_dispatch("add", output, {a, b}, params);
        }
        auto blocked = first.record();
        snapshot.synchronize();
        check(snapshot.ready() && !blocked.ready() && !first.ready(), "Events did not retain independent queue snapshots");
        {
            StreamScope scope(second);
            backend_dispatch("add", independent, {a, b}, params);
        }
        auto free = second.record();
        free.synchronize();
        check(free.ready() && !blocked.ready(), "Independent stream waited for the blocked stream");
        second.wait(blocked);
        auto explicit_dependency = second.record();
        check(!explicit_dependency.ready(), "Explicit event dependency was ignored");
        {
            StreamScope scope(third);
            backend_dispatch("add", dependent, {output, b}, params);
        }
        auto automatic_dependency = third.record();
        check(!automatic_dependency.ready(), "Buffer producer dependency was ignored");
        first.synchronize(); second.synchronize(); third.synchronize();
    }
    check(backend_download(output) == std::vector<float>(16, 3), "Producer output differs");
    check(backend_download(dependent) == std::vector<float>(16, 5), "Cross-stream consumer differs");
    {
        Gate gate(first);
        {
            StreamScope scope(first);
            gate.enqueue();
            backend_dispatch("add", output, {a, b}, params);
        }
        std::promise<void> started;
        std::atomic<bool> finished{false}, failed{false};
        std::thread writer([&] {
            started.set_value();
            try {
                StreamScope scope(second);
                backend_upload(a, std::vector<float>(16, 9));
                finished = true;
            } catch (...) { failed = true; }
        });
        started.get_future().wait();
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
        bool waited = !finished;
        writer.join();
        first.synchronize();
        check(waited && !failed, "Host overwrite did not wait for queued readers");
        check(backend_download(output) == std::vector<float>(16, 3), "Queued read observed a later overwrite");
        check(backend_download(a) == std::vector<float>(16, 9), "Host overwrite was lost");
    }
    {
        AllocationPool pool(Backend::CUDA);
        auto recycled = values(1);
        auto address = recycled->raw_pointer();
        Gate gate(first);
        {
            StreamScope scope(first);
            gate.enqueue();
            backend_dispatch("add", output, {recycled, b}, params);
        }
        recycled.reset();
        BufferPtr replacement;
        {
            StreamScope scope(second);
            replacement = backend_allocate(Backend::CUDA, 16, DType::Float32);
        }
        auto cleared = second.record();
        bool pending = !cleared.ready();
        first.synchronize(); second.synchronize();
        check(pending && replacement->raw_pointer() == address && pool.stats().reuses == 1,
              "Cross-stream pool reuse did not preserve its pending-reader fence");
        check(backend_download(output) == std::vector<float>(16, 3), "Recycled storage changed an earlier read");
        check(backend_download(replacement) == std::vector<float>(16, 0), "Recycled tensor was not cleared");
    }
    return checks;
}
