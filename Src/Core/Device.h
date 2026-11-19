#pragma once

#include "Core/Execution.h"

namespace elman {
struct DeviceInfo {
    Backend backend = Backend::CPU;
    int ordinal = -1;
    std::string name;
    size_t total_bytes = 0;
};
std::vector<DeviceInfo> devices(Backend backend);
int current_device(Backend backend);

// Switching a device suspends a stream binding belonging to another device.
// Both the device and original binding are restored during stack unwinding.
class DeviceScope {
public:
    DeviceScope(Backend backend, int ordinal);
    ~DeviceScope();
    DeviceScope(const DeviceScope&) = delete;
    DeviceScope& operator=(const DeviceScope&) = delete;
private:
    Backend backend_;
    int previous_ = -1;
    detail::StreamBinding previous_binding_;
    bool changed_ = false;
};

// Destination is allocated on the current CUDA device. Transfers preserve all
// storage bits and register dependencies on both owners. A bounded host staging
// route is used when peer access is unavailable.
void copy_device_storage(const BufferPtr& destination, const BufferPtr& source);
} // namespace elman
