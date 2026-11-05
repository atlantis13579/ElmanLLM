#include "Backend.h"

#include <array>
#include <atomic>
#include <stdexcept>

namespace elman {
BufferPtr allocate_cpu(size_t, DType);
void upload_cpu(const BufferPtr&, const std::vector<float>&);
std::vector<float> download_cpu(const BufferPtr&);
void dispatch_cpu(const std::string&, const BufferPtr&,
                  const std::vector<BufferPtr>&, const KernelParams&);

#ifdef ELMAN_HAS_CUDA
BufferPtr allocate_cuda(size_t, DType);
void upload_cuda(const BufferPtr&, const std::vector<float>&);
std::vector<float> download_cuda(const BufferPtr&);
void dispatch_cuda(const std::string&, const BufferPtr&,
                   const std::vector<BufferPtr>&, const KernelParams&);
bool available_cuda();
#endif

namespace {
struct BackendOps {
  Backend id;
  const char* name;
  bool (*available)();
  BufferPtr (*allocate)(size_t, DType);
  void (*upload)(const BufferPtr&, const std::vector<float>&);
  std::vector<float> (*download)(const BufferPtr&);
  void (*dispatch)(const std::string&, const BufferPtr&,
                   const std::vector<BufferPtr>&, const KernelParams&);
};

bool available_cpu() {
  return true;
}

// Add a backend by providing its native functions and one entry here.
const std::array<BackendOps, 2> implementations{{
    {Backend::CPU, "cpu", available_cpu, allocate_cpu, upload_cpu, download_cpu,
     dispatch_cpu},
#ifdef ELMAN_HAS_CUDA
    {Backend::CUDA, "cuda", available_cuda, allocate_cuda, upload_cuda,
     download_cuda, dispatch_cuda},
#else
    {Backend::CUDA, "cuda", nullptr, nullptr, nullptr, nullptr, nullptr},
#endif
}};

size_t backend_index(Backend backend) {
  for (size_t index = 0; index < implementations.size(); ++index) {
    if (implementations[index].id == backend) {
      return index;
    }
  }
  throw std::invalid_argument("Invalid backend");
}

const BackendOps& find_backend(Backend backend) {
  return implementations[backend_index(backend)];
}

std::array<std::atomic<uint64_t>, implementations.size()> counts{};
}  // namespace

std::string backend_name(Backend backend) {
  return find_backend(backend).name;
}

Backend parse_backend(const std::string& name) {
  for (const auto& implementation : implementations) {
    if (name == implementation.name) {
      return implementation.id;
    }
  }
  throw std::invalid_argument("Unknown backend: " + name);
}

bool backend_available(Backend backend) {
  const auto& implementation = find_backend(backend);
  return implementation.available && implementation.available();
}

BufferPtr backend_allocate(Backend backend, size_t count, DType dtype) {
  const auto& implementation = find_backend(backend);
  if (!implementation.allocate) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  return implementation.allocate(count, dtype);
}

void backend_upload(const BufferPtr& buffer, const std::vector<float>& values) {
  if (!buffer || values.size() != buffer->count) {
    throw std::invalid_argument("Upload size mismatch");
  }
  const auto& implementation = find_backend(buffer->backend);
  if (!implementation.upload) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  implementation.upload(buffer, values);
}

std::vector<float> backend_download(const BufferPtr& buffer) {
  if (!buffer) {
    throw std::invalid_argument("Undefined buffer");
  }
  const auto& implementation = find_backend(buffer->backend);
  if (!implementation.download) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  return implementation.download(buffer);
}

void backend_dispatch(const std::string& op, const BufferPtr& output,
                      const std::vector<BufferPtr>& inputs,
                      const KernelParams& params) {
  if (!output) {
    throw std::invalid_argument("Undefined output");
  }
  for (const auto& input : inputs) {
    if (input && input->backend != output->backend) {
      throw std::invalid_argument("Mixed-backend operation");
    }
  }
  const auto& implementation = find_backend(output->backend);
  if (!implementation.dispatch) {
    throw std::runtime_error("Backend not compiled: " +
                             std::string(implementation.name));
  }
  implementation.dispatch(op, output, inputs, params);
  ++counts[backend_index(output->backend)];
}

uint64_t backend_dispatch_count(Backend backend) {
  return counts[backend_index(backend)].load();
}
}  // namespace elman
