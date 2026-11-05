#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <unordered_map>

#include "../Core/Backend.h"

namespace elman {
namespace {
struct CPUBuffer final : Buffer {
  std::vector<uint32_t> words;
  CPUBuffer(size_t count, DType dtype)
      : Buffer(Backend::CPU, dtype, count), words(count) {
  }
};
CPUBuffer& cpu_buffer(const BufferPtr& value) {
  auto* result = dynamic_cast<CPUBuffer*>(value.get());
  if (!result) {
    throw std::invalid_argument("Expected a CPU buffer");
  }
  return *result;
}
}  // namespace
using CPULaunch = void (*)(uint32_t*, const uint32_t*, const uint32_t*,
                           const uint32_t*, const KernelParams&);
#define ELMAN_CPU_KERNEL(name)                                          \
  void launch_##name##_cpu(uint32_t*, const uint32_t*, const uint32_t*, \
                           const uint32_t*, const KernelParams&);
#include "../Operators/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL
BufferPtr allocate_cpu(size_t count, DType dtype) {
  return std::make_shared<CPUBuffer>(count, dtype);
}
void upload_cpu(const BufferPtr& buffer, const std::vector<float>& values) {
  auto& destination = cpu_buffer(buffer);
  if (values.size() != destination.count) {
    throw std::invalid_argument("CPU upload size mismatch");
  }
  for (size_t i = 0; i < values.size(); ++i) {
    float value = values[i];
    if (destination.dtype == DType::Float32) {
      std::memcpy(&destination.words[i], &value, 4);
    } else if (destination.dtype == DType::Bool) {
      destination.words[i] = value != 0.0f ? 1u : 0u;
    } else if (std::isnan(value)) {
      destination.words[i] = 0u;
    } else if (value >= 2147483648.0f) {
      destination.words[i] = 2147483647u;
    } else if (value <= -2147483648.0f) {
      destination.words[i] = 2147483648u;
    } else {
      destination.words[i] = uint32_t(int32_t(value));
    }
  }
}
std::vector<float> download_cpu(const BufferPtr& buffer) {
  auto& source = cpu_buffer(buffer);
  std::vector<float> result(source.count);
  for (size_t i = 0; i < source.count; ++i) {
    if (source.dtype == DType::Float32) {
      std::memcpy(&result[i], &source.words[i], 4);
    } else {
      result[i] = float(int32_t(source.words[i]));
    }
  }
  return result;
}
void dispatch_cpu(const std::string& operation, const BufferPtr& output,
                  const std::vector<BufferPtr>& inputs,
                  const KernelParams& params) {
  static const std::unordered_map<std::string, CPULaunch> kernels = {
#define ELMAN_CPU_KERNEL(name) {#name, launch_##name##_cpu},
#include "../Operators/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL
  };
  const auto found = kernels.find(operation);
  if (found == kernels.end()) {
    throw std::invalid_argument("Unknown CPU operator: " + operation);
  }
  auto& destination = cpu_buffer(output);
  if (params.v[0] < 0 || size_t(params.v[0]) != destination.count) {
    throw std::invalid_argument("CPU dispatch size mismatch");
  }
  const uint32_t* sources[3] = {nullptr, nullptr, nullptr};
  if (inputs.size() > 3) {
    throw std::invalid_argument("CPU kernels accept at most three inputs");
  }
  for (size_t j = 0; j < inputs.size(); ++j) {
    if (inputs[j]) {
      sources[j] = cpu_buffer(inputs[j]).words.data();
    }
  }
  found->second(destination.words.data(), sources[0], sources[1], sources[2],
                params);
}
}  // namespace elman
