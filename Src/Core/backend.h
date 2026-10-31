#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace elman {
enum class Backend {
    CPU,
    CUDA,
    Compute,
    Metal
};
enum class DType {
    Float32 = 0,
    Int32 = 1,
    Bool = 2
};
struct Buffer {
    Backend backend;
    DType dtype;
    size_t count;
    uint64_t version = 0;
    Buffer(Backend b, DType d, size_t n) : backend(b), dtype(d), count(n) {}
    virtual ~Buffer() = default;
};
using BufferPtr = std::shared_ptr<Buffer>;
struct KernelParams {
    int v[256]{};
};
BufferPtr backend_allocate(Backend backend, size_t count, DType dtype);
void backend_upload(const BufferPtr& buffer, const std::vector<float>& values);
std::vector<float> backend_download(const BufferPtr& buffer);
void backend_dispatch(const std::string& op, const BufferPtr& output, const std::vector<BufferPtr>& inputs,
                      const KernelParams& params);
bool backend_available(Backend backend);
std::string backend_name(Backend backend);
Backend parse_backend(const std::string& name);
uint64_t backend_dispatch_count(Backend backend);
} // namespace elman
