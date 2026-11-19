#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include "Core/DType.h"

namespace elman {
enum class Backend { CPU, CUDA, Metal };
struct Buffer {
private:
  uint64_t own_version_ = 0;
  std::shared_ptr<Buffer> owner_;
  size_t offset_bytes_ = 0;
protected:
  Buffer(const std::shared_ptr<Buffer>& source, size_t count, size_t offset_bytes)
      : owner_(source->owner_ ? source->owner_ : source), offset_bytes_(source->offset_bytes_ + offset_bytes),
        backend(source->backend), dtype(source->dtype), count(count), bytes(dtype_storage_bytes(dtype, count)),
        version(owner_->version) {}
  size_t storage_offset_bytes() const { return offset_bytes_; }
public:
  Backend backend;
  DType dtype;
  size_t count;
  size_t bytes;
  uint64_t& version;
  Buffer(Backend b, DType d, size_t n) : backend(b), dtype(d), count(n), bytes(dtype_storage_bytes(d, n)), version(own_version_) {
  }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
  Buffer& allocation() { return owner_ ? *owner_ : *this; }
  const Buffer& allocation() const { return owner_ ? *owner_ : *this; }
  virtual ~Buffer() = default;
  virtual int device_ordinal() const { return owner_ ? owner_->device_ordinal() : -1; }
  virtual void* raw_pointer() = 0;
  virtual const void* raw_pointer() const = 0;
};
using BufferPtr = std::shared_ptr<Buffer>;

enum class KernelInput : int { A = 0, B = 1, C = 2, D = 3 };

#ifdef __CUDACC__
#define ELMAN_KERNEL_HOST_DEVICE __host__ __device__
#else
#define ELMAN_KERNEL_HOST_DEVICE
#endif

struct KernelParams {
  // Packed ABI slots shared by CPU and CUDA. Operators use the accessors below.
  static constexpr int kOutputCountSlot = 0;
  static constexpr int kOutputRankSlot = 1;
  static constexpr int kOutputShapeSlot = 2;
  static constexpr int kInputRankSlot = 10;
  static constexpr int kInputSlotWidth = 17;
  static constexpr int kInputShapeOffset = 1;
  static constexpr int kInputStrideOffset = 9;
  static constexpr int kAttributeSlot = 64;
  static constexpr int kAttributeCount = 64;
  static constexpr int kFourthInputSlot = 128;
  static constexpr int kFourthInputDTypeSlot = 251;
  static constexpr int kInputDTypeSlot = 252;
  static constexpr int kOutputDTypeSlot = 255;

  int v[256]{};

  ELMAN_KERNEL_HOST_DEVICE int output_count() const {
    return v[kOutputCountSlot];
  }
  ELMAN_KERNEL_HOST_DEVICE int output_rank() const {
    return v[kOutputRankSlot];
  }
  ELMAN_KERNEL_HOST_DEVICE int output_shape(int axis) const {
    return v[kOutputShapeSlot + axis];
  }
  ELMAN_KERNEL_HOST_DEVICE int input_rank(KernelInput input) const {
    return v[input_slot(input)];
  }
  ELMAN_KERNEL_HOST_DEVICE static int input_slot(KernelInput input) {
    return input == KernelInput::D ? kFourthInputSlot : kInputRankSlot + kInputSlotWidth * static_cast<int>(input);
  }
  ELMAN_KERNEL_HOST_DEVICE static int input_dtype_slot(KernelInput input) {
    return input == KernelInput::D ? kFourthInputDTypeSlot : kInputDTypeSlot + static_cast<int>(input);
  }
  ELMAN_KERNEL_HOST_DEVICE int input_shape(KernelInput input, int axis) const {
    return v[input_slot(input) + kInputShapeOffset + axis];
  }
  ELMAN_KERNEL_HOST_DEVICE int input_stride(KernelInput input, int axis) const {
    return v[input_slot(input) + kInputStrideOffset + axis];
  }
  ELMAN_KERNEL_HOST_DEVICE int attribute(int offset) const {
    return v[kAttributeSlot + offset];
  }
  ELMAN_KERNEL_HOST_DEVICE int input_dtype(KernelInput input) const {
    return v[input_dtype_slot(input)];
  }
  ELMAN_KERNEL_HOST_DEVICE int output_dtype() const {
    return v[kOutputDTypeSlot];
  }
};

#undef ELMAN_KERNEL_HOST_DEVICE
BufferPtr backend_allocate(Backend backend, size_t count, DType dtype);
// Share a byte-aligned contiguous element range, allocation lifetime and version.
BufferPtr backend_view(const BufferPtr& source, size_t offset, size_t count);
void backend_upload(const BufferPtr& buffer, const std::vector<float>& values);
std::vector<float> backend_download(const BufferPtr& buffer);
void backend_upload_int(const BufferPtr& buffer, const std::vector<int32_t>& values);
std::vector<int32_t> backend_download_int(const BufferPtr& buffer);
void backend_upload_int64(const BufferPtr& buffer, const std::vector<int64_t>& values);
std::vector<int64_t> backend_download_int64(const BufferPtr& buffer);
void backend_upload_raw(const BufferPtr& buffer, const std::vector<uint8_t>& bytes);
std::vector<uint8_t> backend_download_raw(const BufferPtr& buffer);
void backend_dispatch(const std::string& op, const BufferPtr& output,
                      const std::vector<BufferPtr>& inputs,
                      const KernelParams& params);
bool backend_available(Backend backend);
std::string backend_name(Backend backend);
Backend parse_backend(const std::string& name);
uint64_t backend_dispatch_count(Backend backend);
}  // namespace elman
