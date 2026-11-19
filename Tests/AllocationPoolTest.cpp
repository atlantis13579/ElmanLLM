#include "Core/AllocationPool.h"
#include "Core/Graph.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>

namespace {
using namespace elman;
size_t checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
void ownership(Backend backend) {
    auto before = memory_stats(backend).live_bytes();
    BufferPtr escaped;
    std::weak_ptr<Buffer> old;
    {
        AllocationPool pool(backend);
        auto first = backend_allocate(backend, 16, DType::Float32);
        auto pointer = first->raw_pointer();
        backend_upload(first, std::vector<float>(16, 7));
        first->version = 99;
        old = first;
        first.reset();
        check(old.expired(), "Cache kept an old lease alive");
        check(memory_stats(backend).cached_bytes == 64, "Idle allocation was not charged as cached");
        auto second = backend_allocate(backend, 16, DType::Float32);
        check(second->raw_pointer() == pointer && second->version == 0, "Reused storage retained its version");
        check(backend_download(second) == std::vector<float>(16, 0), "Reused tensor storage was not cleared");
        auto alias = second;
        second.reset();
        auto third = backend_allocate(backend, 16, DType::Float32);
        check(third->raw_pointer() != pointer, "Live alias was recycled");
        escaped = std::move(alias);
        third.reset();
        pool.clear();
        check(pool.stats().cached_bytes == 0 && pool.stats().reserved_bytes == 64, "Clear released live storage");
        check(pool.stats().allocations == 2 && pool.stats().reuses == 1, "Pool allocation counters differ");
    }
    check(memory_stats(backend).live_bytes() == before + 64, "Escaped lease did not survive its scope");
    check(backend_download(escaped) == std::vector<float>(16, 0), "Escaped lease lost contents");
    std::thread release([buffer = std::move(escaped)]() mutable { buffer.reset(); });
    release.join();
    check(memory_stats(backend).live_bytes() == before, "Cross-thread release leaked closed pool storage");
}
void formats(Backend backend) {
    auto before = memory_stats(backend).live_bytes();
    {
        AllocationPool pool(backend, before + 4096);
        for (auto dtype : all_dtypes()) for (size_t count : {size_t(0), size_t(1), size_t(3), size_t(17)}) {
            auto first = backend_allocate(backend, count, dtype);
            backend_upload_raw(first, std::vector<uint8_t>(first->bytes, 0xa5));
            first.reset();
            auto reuses = pool.stats().reuses;
            auto second = backend_allocate(backend, count, dtype);
            check(pool.stats().reuses == reuses + 1, "Exact dtype/size allocation was not reused");
            check(backend_download_raw(second) == std::vector<uint8_t>(second->bytes, 0), "Raw dtype bytes survived tensor reuse");
        }
    }
    check(memory_stats(backend).live_bytes() == before, "Format cache leaked at scope exit");
}
void limits(Backend backend) {
    auto before = memory_stats(backend).live_bytes();
    {
        MemoryLimit limit(backend, before + 128);
        AllocationPool pool(backend);
        auto a = backend_allocate(backend, 16, DType::Float32);
        a.reset();
        auto b = backend_allocate(backend, 24, DType::Float32);
        check(pool.stats().evictions == 1 && pool.stats().reserved_bytes == 96, "Hard ceiling did not evict cached storage");
        b.reset();
        // Nonpooled host/vendor reservations must also be able to reclaim cache.
        MemoryReservation workspace(backend, 128);
        check(pool.stats().reserved_bytes == 0 && memory_stats(backend).cached_bytes == 0, "Workspace reservation did not reclaim cache");
    }
    {
        AllocationPool pool(backend, before + 128);
        backend_allocate(backend, 16, DType::Float32);
        auto larger = backend_allocate(backend, 24, DType::Float32);
        check(pool.stats().evictions == 1, "Soft target did not trim idle storage");
        larger.reset();
        auto beyond = backend_allocate(backend, 64, DType::Float32);
        check(beyond->bytes == 256, "Cache target became an unintended hard allocation limit");
        beyond.reset();
        check(pool.stats().cached_bytes == 0, "Oversized live allocation was retained");
    }
    check(memory_stats(backend).live_bytes() == before, "Reclamation left allocations live");
    {
        AllocationPool pool(backend);
        backend_allocate(backend, 16, DType::Float32);
        bool failed = false;
        try { backend_allocate(backend, std::numeric_limits<size_t>::max() / 16, DType::Float32); }
        catch (const std::exception&) { failed = true; }
        check(failed && pool.stats().cached_bytes == 0 && pool.stats().evictions == 1,
              "Physical allocation failure did not reclaim idle cache before retry");
        auto recovered = backend_allocate(backend, 16, DType::Float32);
        check(backend_download(recovered) == std::vector<float>(16, 0), "Pool did not recover after physical OOM");
    }
    {
        AllocationPool disabled(backend, 0, false);
        backend_allocate(backend, 16, DType::Float32);
        backend_allocate(backend, 16, DType::Float32);
        check(disabled.stats().allocations == 2 && disabled.stats().reuses == 0 && disabled.stats().cached_bytes == 0,
              "Disabled pool retained or reused storage");
    }
}
void nesting(Backend backend) {
    auto before = memory_stats(backend).live_bytes();
    {
        AllocationPool outer(backend);
        auto first = backend_allocate(backend, 16, DType::Float32);
        first.reset();
        {
            AllocationPool inner(backend);
            auto second = backend_allocate(backend, 16, DType::Float32);
            check(inner.stats().allocations == 1 && outer.stats().reuses == 0, "Nested scopes shared allocation routing");
            second.reset();
            std::thread release([value = backend_allocate(backend, 16, DType::Float32)]() mutable { value.reset(); });
            release.join();
            check(inner.stats().cached_bytes == 64, "Cross-thread return was not cached safely");
        }
        auto again = backend_allocate(backend, 16, DType::Float32);
        check(outer.stats().reuses == 1, "Nested scope did not restore its parent");
    }
    check(memory_stats(backend).live_bytes() == before, "Nested scopes leaked");
    std::atomic<bool> failed{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < 4; ++i) threads.emplace_back([&] {
        try {
            AllocationPool pool(backend);
            for (int j = 0; j < 20; ++j) backend_allocate(backend, 16, DType::Float32);
            if (pool.stats().allocations != 1 || pool.stats().reuses != 19) failed = true;
        } catch (...) { failed = true; }
    });
    for (auto& thread : threads) thread.join();
    check(!failed && memory_stats(backend).live_bytes() == before, "Thread-local pools interfered");
}
void workspaces_and_gradients(Backend backend) {
    auto before = memory_stats(backend).live_bytes();
    {
        AllocationPool pool(backend);
        auto first = backend_workspace(backend, 64);
        auto pointer = first->raw_pointer();
        first.reset();
        auto second = backend_workspace(backend, 64);
        check(second->raw_pointer() == pointer && memory_stats(backend).workspace_bytes >= 64, "Workspace did not reuse its charge");
        auto tensor = backend_allocate(backend, 64, DType::UInt8);
        check(tensor->raw_pointer() != pointer, "Workspace and tensor leases aliased");
        for (int repeat = 0; repeat < 3; ++repeat) {
            auto x = Tensor::from_data({3}, {1, -2, 3}, backend, true);
            auto loss = sum(mul(x, x));
            loss.backward();
            check(x.grad() == std::vector<float>({2, -4, 6}), "Reused autograd storage changed gradients");
        }
        auto x = Tensor::from_data({2}, {1, 2}, backend, true);
        auto loss = sum(mul(x, x));
        x.assign({3, 4});
        bool rejected = false;
        try { loss.backward(); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Pooling bypassed saved tensor version checks");
    }
    check(memory_stats(backend).live_bytes() == before, "Workspace or gradient cache leaked");
    try {
        AllocationPool pool(backend);
        backend_allocate(backend, 16, DType::Float32);
        throw std::runtime_error("unwind pool");
    } catch (const std::runtime_error&) {}
    check(memory_stats(backend).live_bytes() == before, "Exception did not release idle storage");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_pool_tests cpu|cuda");
        auto backend = elman::parse_backend(argv[1]);
        if (!elman::backend_available(backend)) return 77;
        ownership(backend); formats(backend); limits(backend); nesting(backend); workspaces_and_gradients(backend);
        std::cout << "Allocation pool: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
