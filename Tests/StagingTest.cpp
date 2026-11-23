#include "Core/Execution.h"
#include "Core/AllocationPool.h"
#include "Core/Tensor.h"
#include <algorithm>
#include <iostream>
#include <limits>

#ifdef ELMAN_HAS_CUDA
int cuda_staging_ordering_checks();
#endif
namespace {
using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
void bindings(Backend backend) {
    ExecutionStream first(backend), second(backend);
    check(host_staging_memory_bound() == 4 * 1024 * 1024 + 256 * 128, "Staging memory bound differs");
    check(!host_staging_memory_bound({0, 256}) && !host_staging_memory_bound({4096, 0}), "Disabled staging has a bound");
    bool rejected = false;
    try { ExecutionStream invalid(backend, {4096, std::numeric_limits<size_t>::max()}); }
    catch (const std::overflow_error&) { rejected = true; }
    check(rejected, "Overflowing descriptor bound accepted");
    check(!pinned_staging_enabled(backend), "Unexpected default staging binding");
    {
        StreamScope ordinary(first);
        check(!pinned_staging_enabled(backend), "Ordinary scope enabled staging");
        {
            StreamScope enabled(first, true, true);
            check(pinned_staging_enabled(backend) == (backend == Backend::CUDA), "Explicit staging binding missing");
            try {
                StreamScope synchronous(second, false, true);
                check(!pinned_staging_enabled(backend), "Synchronous scope enabled staging");
                throw 7;
            } catch (int) {}
            check(pinned_staging_enabled(backend) == (backend == Backend::CUDA), "Unwind lost staging binding");
        }
        check(!pinned_staging_enabled(backend), "Nested scope leaked staging binding");
    }
    first.trim_host_staging();
    check(first.host_staging_stats().allocated_bytes == 0, "Empty stream allocated staging");
}
void storage(Backend backend) {
    size_t host_before = memory_stats(Backend::CPU).live_bytes();
    size_t device_before = memory_stats(backend).live_bytes();
    for (bool reuse : {false, true}) for (bool asynchronous : {false, true}) for (bool pinned : {false, true}) {
        ExecutionStream stream(backend, {16384, 4});
        StreamScope scope(stream, asynchronous, pinned);
        AllocationPool pool(backend, 0, reuse);
        for (elman::DType dtype : all_dtypes()) for (size_t count : {size_t(0), size_t(1), size_t(17), size_t(257)}) {
            BufferPtr buffer = backend_allocate(backend, count, dtype);
            std::vector<uint8_t> expected(buffer->bytes);
            for (size_t i = 0; i < expected.size(); ++i) expected[i] = uint8_t(i * 73 + 11);
            std::vector<uint8_t> caller = expected;
            backend_upload_raw(buffer, caller);
            std::fill(caller.begin(), caller.end(), uint8_t(0));
            check(backend_download_raw(buffer) == expected, "Staging changed raw storage bytes");
        }
        Tensor a = Tensor::from_data({4}, {1, -2, 3, -4}, backend);
        check(mul(a, a).data() == std::vector<float>({1, 4, 9, 16}), "Staged arithmetic input differs");
        stream.synchronize();
        HostStagingStats stats = stream.host_staging_stats();
        bool active = backend == Backend::CUDA && asynchronous && pinned;
        check((stats.uploads > 0) == active && (stats.downloads > 0) == active, "Staging selection differs");
        check(stats.allocated_bytes <= 16384 && stats.buffers <= 4 && stats.pending_bytes == 0, "Staging capacity or retirement differs");
        if (active) check(stats.reuses > 0 && stats.cached_bytes == stats.allocated_bytes, "Completed staging was not reusable");
        stream.trim_host_staging();
        stats = stream.host_staging_stats();
        check(stats.allocated_bytes == 0 && stats.metadata_bytes == 0 && stats.buffers == 0, "Trim retained completed staging");
    }
    check(memory_stats(Backend::CPU).live_bytes() == host_before, "Staging leaked host memory");
    check(memory_stats(backend).live_bytes() == device_before, "Staging leaked device memory");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_staging_tests cpu|cuda");
        Backend backend = elman::parse_backend(argv[1]);
        if (!elman::backend_available(backend)) return 77;
        bindings(backend); storage(backend);
#ifdef ELMAN_HAS_CUDA
        if (backend == elman::Backend::CUDA) checks += cuda_staging_ordering_checks();
#endif
        std::cout << "Host staging: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
