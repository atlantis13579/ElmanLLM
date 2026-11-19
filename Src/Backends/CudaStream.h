#pragma once
#include <cuda_runtime.h>

namespace elman {
cudaStream_t current_cuda_stream();
void check_cuda_execution(cudaError_t status, const char* operation);
} // namespace elman
