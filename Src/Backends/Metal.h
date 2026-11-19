#pragma once

#include "Core/Device.h"
#include "Core/Memory.h"

namespace elman {
bool available_metal();
int allocation_device_metal();
std::vector<DeviceInfo> devices_metal();
MemoryInfo memory_info_metal();
BufferPtr allocate_metal(size_t count, DType dtype);
void clear_metal(const BufferPtr& buffer);
void upload_metal(const BufferPtr& buffer, const std::vector<float>& values);
std::vector<float> download_metal(const BufferPtr& buffer);
void upload_int_metal(const BufferPtr& buffer, const std::vector<int32_t>& values);
std::vector<int32_t> download_int_metal(const BufferPtr& buffer);
void upload_raw_metal(const BufferPtr& buffer, const std::vector<uint8_t>& values);
std::vector<uint8_t> download_raw_metal(const BufferPtr& buffer);
void dispatch_metal(const std::string& operation, const BufferPtr& output,
                    const std::vector<BufferPtr>& inputs, const KernelParams& params);
namespace detail {
std::shared_ptr<StreamState> create_metal_stream();
}
} // namespace elman
