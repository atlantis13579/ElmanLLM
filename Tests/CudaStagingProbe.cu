#include "Core/Execution.h"
#include "Core/Memory.h"
#include "Backends/CudaStream.h"
#include <chrono>
#include <thread>

namespace {
using namespace elman;
struct Gate {
    ExecutionStream& owner;
    explicit Gate(ExecutionStream& stream) : owner(stream) {}
    ~Gate() { try { owner.synchronize(); } catch (...) {} }
    static void CUDART_CB wait(void*) { std::this_thread::sleep_for(std::chrono::milliseconds(500)); }
    void enqueue() { check_cuda_execution(cudaLaunchHostFunc(current_cuda_stream(), wait, nullptr), "Staging test gate"); }
};
void warm(ExecutionStream& stream, const BufferPtr& buffer) {
    StreamScope scope(stream, true, true);
    backend_upload_raw(buffer, std::vector<uint8_t>(buffer->bytes, 3));
    stream.synchronize();
}
KernelParams copy_bytes() {
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = 16;
    p.v[KernelParams::kOutputRankSlot] = 1;
    p.v[KernelParams::kOutputShapeSlot] = 16;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int32);
    p.v[KernelParams::kInputRankSlot] = 1;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = 16;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 1;
    p.v[KernelParams::kInputDTypeSlot] = int(DType::Int32);
    return p;
}
}
int cuda_staging_ordering_checks() {
    int checks = 0;
    auto check = [&](bool valid, const char* message) {
        ++checks;
        if (!valid) throw std::runtime_error(message);
    };
    auto before = memory_stats(Backend::CPU).live_bytes();
    {
        ExecutionStream first(Backend::CUDA, {4096, 1}), second(Backend::CUDA, {4096, 1});
        ExecutionStream reader(Backend::CUDA);
        auto buffer = backend_allocate(Backend::CUDA, 16, DType::Int32);
        auto observed = backend_allocate(Backend::CUDA, 16, DType::Int32);
        warm(first, buffer); warm(second, buffer);
        backend_dispatch("copy", observed, {buffer}, copy_bytes());
        Gate gate(first);
        ExecutionEvent pending;
        {
            StreamScope scope(first, true, true);
            gate.enqueue();
            std::vector<uint8_t> caller(64, 17);
            backend_upload_raw(buffer, caller);
            std::fill(caller.begin(), caller.end(), uint8_t(99));
            pending = first.record();
            check(!pending.ready(), "Staged upload waited for its device queue");
        }
        {
            StreamScope scope(reader);
            backend_dispatch("copy", observed, {buffer}, copy_bytes());
            check(!reader.record().ready(), "Queued reader ignored a staged producer");
        }
        // The replacement writer drops the tensor's reference to the first event.
        // The originating stream must still own the earlier pinned source.
        {
            StreamScope scope(second, true, true);
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 23));
            check(!second.record().ready(), "Staged overwrite ignored its prior writer");
        }
        first.trim_host_staging();
        check(first.host_staging_stats().pending_bytes == 4096, "Replacing the writer released pending host storage");
        second.synchronize(); first.synchronize();
        check(backend_download_raw(observed) == std::vector<uint8_t>(64, 17), "Queued reader observed mutated caller bytes or a later overwrite");
        check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 23), "Cross-stream overwrite differs");
        {
            StreamScope scope(first, true, true);
            gate.enqueue();
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 31));
            auto uploaded = first.record();
            check(!uploaded.ready(), "Prewarmed upload lost asynchronous behavior");
            // One slot forces bounded backpressure before reusing pending storage.
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 37));
            check(uploaded.ready() && first.host_staging_stats().waits > 0, "Full staging pool did not wait for reuse");
            check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 37), "Bounded reuse corrupted an upload");
        }
        first.synchronize(); second.synchronize();
        first.trim_host_staging(); second.trim_host_staging();
        check(memory_stats(Backend::CPU).live_bytes() == before, "Trim retained host reservations");
        // Host ceilings make staging optional, while the pageable path remains correct.
        {
            MemoryLimit tight(Backend::CPU, before);
            StreamScope scope(first, true, true);
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 41));
            check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 41), "Host-limit fallback changed bytes");
        }
        check(first.host_staging_stats().fallbacks >= 2, "Host-limit fallback was not counted");
        auto large = backend_allocate(Backend::CUDA, 8192, DType::UInt8);
        {
            StreamScope scope(first, true, true);
            gate.enqueue();
            backend_upload_raw(large, std::vector<uint8_t>(8192, 43));
            check(first.ready(), "Oversized pageable upload omitted its host barrier");
            check(backend_download_raw(large) == std::vector<uint8_t>(8192, 43), "Oversized fallback changed bytes");
        }
        check(first.host_staging_stats().fallbacks >= 4, "Oversized fallback was not counted");
        bool rejected = false;
        try {
            StreamScope scope(first, true, true);
            backend_upload_raw(buffer, std::vector<uint8_t>(63));
        } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected, "Invalid staged byte count accepted");
        // A download on another queue observes the producer and returns ready bytes.
        warm(first, buffer);
        {
            StreamScope scope(first, true, true);
            gate.enqueue();
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 47));
        }
        {
            StreamScope scope(second, true, true);
            check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 47), "Cross-stream staged download missed its producer");
        }
        first.synchronize();
    }
    check(memory_stats(Backend::CPU).live_bytes() == before, "Stream destruction leaked staging");
    {
        auto buffer = backend_allocate(Backend::CUDA, 64, DType::UInt8);
        {
            ExecutionStream stream(Backend::CUDA, {4096, 1});
            warm(stream, buffer);
            StreamScope scope(stream, true, true);
            check_cuda_execution(cudaLaunchHostFunc(current_cuda_stream(), Gate::wait, nullptr), "Staging destruction gate");
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 53));
        }
        check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 53), "Stream destruction released an in-flight upload");
    }
    check(memory_stats(Backend::CPU).live_bytes() == before, "Pending destruction leaked staging");
    for (auto bounds : {HostStagingOptions{4096, 4}, HostStagingOptions{16384, 1}}) {
        ExecutionStream stream(Backend::CUDA, bounds);
        auto buffer = backend_allocate(Backend::CUDA, 64, DType::UInt8);
        warm(stream, buffer);
        Gate gate(stream);
        StreamScope scope(stream, true, true);
        gate.enqueue();
        backend_upload_raw(buffer, std::vector<uint8_t>(64, 59));
        auto first_copy = stream.record();
        backend_upload_raw(buffer, std::vector<uint8_t>(64, 61));
        auto stats = stream.host_staging_stats();
        check(first_copy.ready() && stats.waits > 0, "Independent byte/slot bounds did not apply backpressure");
        check(stats.allocated_bytes <= bounds.capacity_bytes && stats.buffers <= bounds.max_buffers, "Pool exceeded its independent bounds");
        check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 61), "Bounded pool changed caller bytes");
    }
    for (auto bounds : {HostStagingOptions{0, 0}, HostStagingOptions{4095, 1}, HostStagingOptions{12288, 1}}) {
        ExecutionStream stream(Backend::CUDA, bounds);
        StreamScope scope(stream, true, true);
        auto buffer = backend_allocate(Backend::CUDA, 8193, DType::UInt8);
        backend_upload_raw(buffer, std::vector<uint8_t>(8193, 67));
        check(backend_download_raw(buffer) == std::vector<uint8_t>(8193, 67), "Non-power-of-two capacity changed bytes");
        stream.synchronize();
        auto stats = stream.host_staging_stats();
        if (bounds.capacity_bytes == 12288) check(stats.allocated_bytes == 12288 && stats.uploads == 1, "Page rounding rejected fitting capacity");
        else check(stats.fallbacks == 2 && stats.allocated_bytes == 0, "Disabled/small capacity did not fall back");
    }
    {
        ExecutionStream stream(Backend::CUDA, {4096, 1});
        auto buffer = backend_allocate(Backend::CUDA, 64, DType::UInt8);
        {
            MemoryLimit metadata_only(Backend::CPU, before + 128);
            StreamScope scope(stream, true, true);
            backend_upload_raw(buffer, std::vector<uint8_t>(64, 71));
            check(backend_download_raw(buffer) == std::vector<uint8_t>(64, 71), "Payload allocation rejection lost bytes");
        }
        check(stream.host_staging_stats().allocated_bytes == 0 && stream.host_staging_stats().metadata_bytes == 128,
              "Rejected payload corrupted descriptor accounting");
        stream.trim_host_staging();
    }
    check(memory_stats(Backend::CPU).live_bytes() == before, "Bounded/fallback paths leaked staging");
    return checks;
}
