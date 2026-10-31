#include "backend.h"
#include <array>
#include <atomic>
#include <stdexcept>

namespace elman {
#define DECLARE_BACKEND(suffix)                                                                                        \
    BufferPtr allocate_##suffix(size_t, DType);                                                                        \
    void upload_##suffix(const BufferPtr&, const std::vector<float>&);                                                 \
    std::vector<float> download_##suffix(const BufferPtr&);                                                            \
    void dispatch_##suffix(const std::string&, const BufferPtr&, const std::vector<BufferPtr>&, const KernelParams&);  \
    bool available_##suffix();
DECLARE_BACKEND(cpu)
#ifdef ELMAN_HAS_CUDA
DECLARE_BACKEND(cuda)
#endif
#ifdef ELMAN_HAS_COMPUTE
DECLARE_BACKEND(compute)
#endif
#ifdef ELMAN_HAS_METAL
DECLARE_BACKEND(metal)
#endif
#undef DECLARE_BACKEND
static std::array<std::atomic<uint64_t>, 4> counts{};
std::string backend_name(Backend b) {
    switch (b) {
    case Backend::CPU:
        return "cpu";
    case Backend::CUDA:
        return "cuda";
    case Backend::Compute:
        return "compute";
    case Backend::Metal:
        return "metal";
    }
    throw std::invalid_argument("Invalid backend");
}
Backend parse_backend(const std::string& s) {
    if (s == "cpu")
        return Backend::CPU;
    if (s == "cuda")
        return Backend::CUDA;
    if (s == "compute" || s == "d3d11")
        return Backend::Compute;
    if (s == "metal")
        return Backend::Metal;
    throw std::invalid_argument("Unknown backend: " + s);
}
bool backend_available(Backend b) {
    switch (b) {
    case Backend::CPU:
        return true;
#ifdef ELMAN_HAS_CUDA
    case Backend::CUDA:
        return available_cuda();
#endif
#ifdef ELMAN_HAS_COMPUTE
    case Backend::Compute:
        return available_compute();
#endif
#ifdef ELMAN_HAS_METAL
    case Backend::Metal:
        return available_metal();
#endif
    default:
        return false;
    }
}
#define BACKEND_SWITCH(call)                                                                                           \
    switch (b) {                                                                                                       \
    case Backend::CPU:                                                                                                 \
        return call(cpu);                                                                                              \
        ELMAN_CUDA_CASE(call)                                                                                          \
        ELMAN_COMPUTE_CASE(call) ELMAN_METAL_CASE(call) default                                                        \
            : throw std::runtime_error("Backend not compiled: " + backend_name(b));                                    \
    }
#ifdef ELMAN_HAS_CUDA
#define ELMAN_CUDA_CASE(call)                                                                                          \
    case Backend::CUDA:                                                                                                \
        return call(cuda);
#else
#define ELMAN_CUDA_CASE(call)
#endif
#ifdef ELMAN_HAS_COMPUTE
#define ELMAN_COMPUTE_CASE(call)                                                                                       \
    case Backend::Compute:                                                                                             \
        return call(compute);
#else
#define ELMAN_COMPUTE_CASE(call)
#endif
#ifdef ELMAN_HAS_METAL
#define ELMAN_METAL_CASE(call)                                                                                         \
    case Backend::Metal:                                                                                               \
        return call(metal);
#else
#define ELMAN_METAL_CASE(call)
#endif
BufferPtr backend_allocate(Backend b, size_t count, DType dtype) {
#define CALL(s) allocate_##s(count, dtype)
    BACKEND_SWITCH(CALL)
#undef CALL
}
void backend_upload(const BufferPtr& buffer, const std::vector<float>& values) {
    if (!buffer || values.size() != buffer->count)
        throw std::invalid_argument("Upload size mismatch");
    Backend b = buffer->backend;
#define CALL(s) upload_##s(buffer, values)
    BACKEND_SWITCH(CALL)
#undef CALL
}
std::vector<float> backend_download(const BufferPtr& buffer) {
    if (!buffer)
        throw std::invalid_argument("Undefined buffer");
    Backend b = buffer->backend;
#define CALL(s) download_##s(buffer)
    BACKEND_SWITCH(CALL)
#undef CALL
}
void backend_dispatch(const std::string& op, const BufferPtr& out, const std::vector<BufferPtr>& in,
                      const KernelParams& p) {
    if (!out)
        throw std::invalid_argument("Undefined output");
    for (const auto& a : in)
        if (a && a->backend != out->backend)
            throw std::invalid_argument("Mixed-backend operation");
    Backend b = out->backend;
    ++counts[static_cast<size_t>(b)];
#define CALL(s) dispatch_##s(op, out, in, p)
    BACKEND_SWITCH(CALL)
#undef CALL
}
uint64_t backend_dispatch_count(Backend b) {
    return counts.at(static_cast<size_t>(b)).load();
}
} // namespace elman
