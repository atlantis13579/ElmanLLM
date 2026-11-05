#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace elman {
enum class Backend { CPU, CUDA };
enum class DType { Float32 = 0, Int32 = 1, Bool = 2 };
struct Buffer {
  Backend backend;
  DType dtype;
  size_t count;
  uint64_t version = 0;
  Buffer(Backend b, DType d, size_t n) : backend(b), dtype(d), count(n) {
  }
  virtual ~Buffer() = default;
};
using BufferPtr = std::shared_ptr<Buffer>;

enum class KernelInput : int { A = 0, B = 1, C = 2 };

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
    return v[kInputRankSlot + kInputSlotWidth * static_cast<int>(input)];
  }
  ELMAN_KERNEL_HOST_DEVICE int input_shape(KernelInput input, int axis) const {
    return v[kInputRankSlot + kInputSlotWidth * static_cast<int>(input) +
             kInputShapeOffset + axis];
  }
  ELMAN_KERNEL_HOST_DEVICE int input_stride(KernelInput input, int axis) const {
    return v[kInputRankSlot + kInputSlotWidth * static_cast<int>(input) +
             kInputStrideOffset + axis];
  }
  ELMAN_KERNEL_HOST_DEVICE int attribute(int offset) const {
    return v[kAttributeSlot + offset];
  }
  ELMAN_KERNEL_HOST_DEVICE int input_dtype(KernelInput input) const {
    return v[kInputDTypeSlot + static_cast<int>(input)];
  }
  ELMAN_KERNEL_HOST_DEVICE int output_dtype() const {
    return v[kOutputDTypeSlot];
  }
};

#undef ELMAN_KERNEL_HOST_DEVICE
BufferPtr backend_allocate(Backend backend, size_t count, DType dtype);
void backend_upload(const BufferPtr& buffer, const std::vector<float>& values);
std::vector<float> backend_download(const BufferPtr& buffer);
void backend_dispatch(const std::string& op, const BufferPtr& output,
                      const std::vector<BufferPtr>& inputs,
                      const KernelParams& params);
bool backend_available(Backend backend);
std::string backend_name(Backend backend);
Backend parse_backend(const std::string& name);
uint64_t backend_dispatch_count(Backend backend);
}  // namespace elman
