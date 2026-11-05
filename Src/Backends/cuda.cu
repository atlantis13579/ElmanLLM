#include <cuda_runtime.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "Backends/HostWords.h"
#include "Core/Backend.h"

namespace elman {
namespace {
void cuda_check(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(result));
  }
}

struct CudaBuffer final : Buffer {
  std::uint32_t* data = nullptr;
  CudaBuffer(std::size_t size, DType type) : Buffer(Backend::CUDA, type, size) {
    if (size >
        std::numeric_limits<std::size_t>::max() / sizeof(std::uint32_t)) {
      throw std::runtime_error("CUDA allocation size overflow");
    }
    cuda_check(
        cudaMalloc(reinterpret_cast<void**>(&data),
                   std::max(size, std::size_t(1)) * sizeof(std::uint32_t)),
        "cudaMalloc");
    try {
      cuda_check(
          cudaMemset(data, 0,
                     std::max(size, std::size_t(1)) * sizeof(std::uint32_t)),
          "cudaMemset");
    } catch (...) {
      cudaFree(data);
      throw;
    }
  }
  ~CudaBuffer() override {
    if (data) {
      cudaFree(data);
    }
  }
};

CudaBuffer& checked_cuda(const BufferPtr& buffer) {
  auto* value = dynamic_cast<CudaBuffer*>(buffer.get());
  if (!value) {
    throw std::runtime_error("Expected a CUDA buffer");
  }
  return *value;
}
}  // namespace

using CudaLaunch = void (*)(std::uint32_t*, const std::uint32_t*,
                            const std::uint32_t*, const std::uint32_t*,
                            const KernelParams&);
#define ELMAN_CUDA_KERNEL(name)                                         \
  void launch_##name##_cuda(std::uint32_t*, const std::uint32_t*,       \
                            const std::uint32_t*, const std::uint32_t*, \
                            const KernelParams&);
#include "Operators/cuda_dispatch.inc"
#undef ELMAN_CUDA_KERNEL

bool available_cuda() {
  int devices = 0;
  const cudaError_t error = cudaGetDeviceCount(&devices);
  if (error != cudaSuccess) {
    cudaGetLastError();
    return false;
  }
  return devices > 0;
}

BufferPtr allocate_cuda(std::size_t count, DType dtype) {
  return std::make_shared<CudaBuffer>(count, dtype);
}

void upload_cuda(const BufferPtr& buffer, const std::vector<float>& values) {
  auto& target = checked_cuda(buffer);
  if (values.size() != target.count) {
    throw std::runtime_error("CUDA upload element count mismatch");
  }
  const auto words = encode_words(values, target.dtype);
  if (!words.empty()) {
    cuda_check(cudaMemcpy(target.data, words.data(),
                          words.size() * sizeof(std::uint32_t),
                          cudaMemcpyHostToDevice),
               "CUDA upload");
  }
}

std::vector<float> download_cuda(const BufferPtr& buffer) {
  auto& source = checked_cuda(buffer);
  std::vector<std::uint32_t> words(source.count);
  if (!words.empty()) {
    cuda_check(cudaMemcpy(words.data(), source.data,
                          words.size() * sizeof(std::uint32_t),
                          cudaMemcpyDeviceToHost),
               "CUDA download");
  }
  return decode_words(words, source.dtype);
}

void dispatch_cuda(const std::string& op, const BufferPtr& output,
                   const std::vector<BufferPtr>& inputs,
                   const KernelParams& params) {
  static const std::unordered_map<std::string, CudaLaunch> kernels = {
#define ELMAN_CUDA_KERNEL(name) {#name, launch_##name##_cuda},
#include "Operators/cuda_dispatch.inc"
#undef ELMAN_CUDA_KERNEL
  };
  auto found = kernels.find(op);
  if (found == kernels.end()) {
    throw std::runtime_error("Unknown CUDA operator: " + op);
  }
  auto& target = checked_cuda(output);
  if (inputs.size() > 3) {
    throw std::runtime_error("CUDA kernels accept at most three inputs");
  }
  if (params.v[0] < 0 ||
      static_cast<std::size_t>(params.v[0]) != target.count) {
    throw std::runtime_error("CUDA dispatch output count mismatch");
  }
  const std::uint32_t* source[3] = {target.data, target.data, target.data};
  for (std::size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i]) {
      source[i] = checked_cuda(inputs[i]).data;
    }
  }
  if (target.count == 0) {
    return;
  }
  found->second(target.data, source[0], source[1], source[2], params);
  cuda_check(cudaGetLastError(), ("CUDA launch " + op).c_str());
  cuda_check(cudaDeviceSynchronize(), ("CUDA execution " + op).c_str());
}
}  // namespace elman
