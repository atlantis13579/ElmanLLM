#include "Core/Device.h"
#include "Core/AllocationPool.h"
#include "Core/Memory.h"
#include "Fusion/Fusion.h"
#include <stdexcept>
#ifdef ELMAN_HAS_METAL
#include "Backends/Metal.h"
#endif

namespace elman {
#ifdef ELMAN_HAS_CUDA
std::vector<DeviceInfo> devices_cuda();
void select_cuda_device(int ordinal);
void copy_cuda_storage(const BufferPtr& destination, const BufferPtr& source);
#endif
std::vector<DeviceInfo> devices(Backend backend) {
    if (backend == Backend::CPU) return {{Backend::CPU, -1, "CPU", memory_info(Backend::CPU).total_bytes}};
    if (backend == Backend::Metal) {
#ifdef ELMAN_HAS_METAL
        return devices_metal();
#else
        return {};
#endif
    }
    if (backend != Backend::CUDA) throw std::invalid_argument("Unknown device backend");
#ifdef ELMAN_HAS_CUDA
    return devices_cuda();
#else
    return {};
#endif
}
int current_device(Backend backend) { return detail::allocation_device(backend); }
DeviceScope::DeviceScope(Backend backend, int ordinal) : backend_(backend) {
    if (backend == Backend::CPU) {
        if (ordinal != -1) throw std::invalid_argument("CPU device ordinal must be -1");
        return;
    }
    if (backend == Backend::Metal) {
        if (ordinal != 0 || !backend_available(backend)) throw std::invalid_argument("Metal device 0 is unavailable");
        return;
    }
    if (backend != Backend::CUDA) throw std::invalid_argument("Unknown device backend");
#ifdef ELMAN_HAS_CUDA
    previous_ = current_device(backend);
    previous_binding_ = detail::stream_binding(backend);
    if (ordinal != previous_) { select_cuda_device(ordinal); changed_ = true; }
    if (previous_binding_.stream && previous_binding_.stream->device != ordinal) {
        detail::stream_binding(backend) = {};
        changed_ = true;
    }
#else
    throw std::runtime_error("CUDA devices are unavailable in this build");
#endif
}
DeviceScope::~DeviceScope() {
#ifdef ELMAN_HAS_CUDA
    if (changed_) {
        detail::stream_binding(backend_) = std::move(previous_binding_);
        try { select_cuda_device(previous_); } catch (...) {}
    }
#endif
}
void copy_device_storage(const BufferPtr& destination, const BufferPtr& source) {
    fusion::access(source);
    fusion::access(destination);
    if (!destination || !source || destination->backend != Backend::CUDA || source->backend != Backend::CUDA ||
        destination->dtype != source->dtype || destination->count != source->count)
        throw std::invalid_argument("CUDA transfer requires matching source/destination storage");
#ifdef ELMAN_HAS_CUDA
    copy_cuda_storage(destination, source);
#else
    throw std::runtime_error("CUDA transfer is unavailable in this build");
#endif
}
} // namespace elman
