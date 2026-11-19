#pragma once
#include "Core/Execution.h"
#include "Backends/CudaStream.h"
#include <map>
#include <mutex>

namespace elman {
// Serializes host submission and access metadata, not device execution.
std::recursive_mutex& cuda_submission_mutex();
uint64_t current_cuda_stream_id();
std::shared_ptr<void>& current_cuda_blas();
void synchronize_cuda_stream();
struct CudaEventState;
struct CudaAccess {
    std::shared_ptr<CudaEventState> writer;
    std::map<uint64_t, std::shared_ptr<CudaEventState>> readers;
};
std::shared_ptr<CudaEventState> record_cuda_completion();
// Empty/oversized transfers or unavailable staging fall back to the host barrier.
std::shared_ptr<CudaEventState> stage_cuda_upload(void* destination, const void* source, size_t bytes);
bool stage_cuda_download(void* destination, const void* source, size_t bytes);
void prepare_cuda_read(const CudaAccess& access);
void prepare_cuda_write(const CudaAccess& access);
void finish_cuda_read(CudaAccess& access, const std::shared_ptr<CudaEventState>& event);
void finish_cuda_write(CudaAccess& access, const std::shared_ptr<CudaEventState>& event);
void retain_cuda_workspace(const BufferPtr& buffer);
} // namespace elman
