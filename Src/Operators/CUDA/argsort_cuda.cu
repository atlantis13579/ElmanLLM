#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "Operators/sort_params.h"
#include "Operators/integer_pointwise.h"
#include "Operators/low_precision_pointwise_math.h"
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <stdexcept>
#include <string>

namespace elman {
namespace {
void check_sort(cudaError_t error) {
  if (error != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("CUDA argsort: ") + cudaGetErrorString(error));
  }
}
struct SortBuffer {
  BufferPtr storage;
  void* data;
  explicit SortBuffer(size_t bytes) : storage(backend_workspace(Backend::CUDA, bytes)), data(storage->raw_pointer()) {}
  SortBuffer(const SortBuffer&) = delete;
  SortBuffer& operator=(const SortBuffer&) = delete;
};
__device__ uint32_t sort_key(const uint32_t* a, int source, const KernelParams& p) {
  uint32_t key;
  if (kernel_integer(p.input_dtype(KernelInput::A))) key = uint32_t(READ_INT_A(source)) ^ 0x80000000u;
  else {
    float value = is_low_precision_float(DType(p.input_dtype(KernelInput::A))) ? low_precision_value(a, source, DType(p.input_dtype(KernelInput::A))) : READ_A(source);
    uint32_t bits = value == 0 ? 0 : __float_as_uint(value);
    // Decoded low precision references use ATen's stable radix ordering,
    // including the sign of a NaN. Other dtypes retain their established order.
    const bool low = is_low_precision_float(DType(p.input_dtype(KernelInput::A)));
    key = !low && isnan(value) ? 0xffffffffu : bits ^ ((bits & 0x80000000u) ? 0xffffffffu : 0x80000000u);
  }
  return p.attribute(operator_attr::sorting::kDescending) ? ~key : key;
}
__global__ void make_sort_keys(uint64_t* keys, int32_t* indices, const uint32_t* a,
                              KernelParams p, int length, int inner) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= p.output_count()) return;
  int group = i / length, column = i % length;
  int source = group / inner * length * inner + group % inner + column * inner;
  const uint32_t key = sort_key(a, source, p);
  keys[i] = (uint64_t(group) << 32) | key;
  indices[i] = column;
}
template<int Items>
__global__ void sort_contiguous_rows(uint32_t* out, const uint32_t* a, int length, KernelParams p) {
  constexpr int threads = 128;
  using Sort = cub::BlockRadixSort<uint32_t, threads, Items, uint32_t>;
  __shared__ typename Sort::TempStorage scratch;
  uint32_t keys[Items], indices[Items];
  const int base = int(blockIdx.x) * length;
  // Blocked input order makes ties stable, including signed zero and NaN.
  #pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int col = int(threadIdx.x) * Items + j;
    keys[j] = col < length ? sort_key(a, base + col, p) : UINT32_MAX;
    indices[j] = col;
  }
  Sort(scratch).SortBlockedToStriped(keys, indices);
  #pragma unroll
  for (int j = 0; j < Items; ++j) {
    const int col = int(threadIdx.x) + threads * j;
    if (col < length) out[base + col] = indices[j];
  }
}
__global__ void store_sort_indices(uint32_t* out, const int32_t* indices, int count, int length, int inner) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i >= count) return;
  int group = i / length, column = i % length;
  int target = group / inner * length * inner + group % inner + column * inner;
  out[target] = uint32_t(indices[i]);
}
__global__ void make_wide_sort_keys(uint64_t* keys, int32_t* indices, int32_t* offsets,
                                      const uint32_t* a, KernelParams p, int length, int inner) {
  const unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= unsigned(p.output_count())) return;
  const int group = int(i) / length, column = int(i) % length;
  const int source = group / inner * length * inner + group % inner + column * inner;
  const int dtype = p.input_dtype(KernelInput::A);
  uint64_t key;
  if (dtype == int(DType::Float64)) {
    const double value = reinterpret_cast<const double*>(a)[source];
    const uint64_t bits = value == 0 ? 0 : uint64_t(__double_as_longlong(value));
    key = isnan(value) ? UINT64_MAX : bits ^ ((bits >> 63) ? UINT64_MAX : uint64_t(1) << 63);
  } else {
    const uint64_t sign = integer_signed(dtype) ? uint64_t(1) << 63 : 0;
    key = integer_reduction_value(a, source, dtype) ^ sign;
  }
  if (p.attribute(operator_attr::sorting::kDescending)) key = ~key;
  keys[i] = key;
  indices[i] = column;
  if (column == 0) offsets[group] = int(i);
  if (i + 1 == unsigned(p.output_count())) offsets[group + 1] = p.output_count();
}
}
void launch_argsort_cuda(uint32_t* out, const uint32_t* a, const uint32_t*,
                         const uint32_t*, const KernelParams& p) {
  int count = p.output_count();
  if (!count) return;
  int axis = p.attribute(operator_attr::sorting::kAxis);
  int length = p.input_shape(KernelInput::A, axis), inner = p.input_stride(KernelInput::A, axis);
  const auto dtype = DType(p.input_dtype(KernelInput::A));
  if (inner == 1 && length > 0 && length <= 1024 && is_compute_float(dtype)) {
    const auto rows = count / length;
    if (length <= 128) sort_contiguous_rows<1><<<rows, 128, 0, current_cuda_stream()>>>(out, a, length, p);
    else if (length <= 512) sort_contiguous_rows<4><<<rows, 128, 0, current_cuda_stream()>>>(out, a, length, p);
    else sort_contiguous_rows<8><<<rows, 128, 0, current_cuda_stream()>>>(out, a, length, p);
    return;
  }
  SortBuffer keys_in(size_t(count) * sizeof(uint64_t)), keys_out(size_t(count) * sizeof(uint64_t));
  SortBuffer ids_in(size_t(count) * sizeof(int32_t)), ids_out(size_t(count) * sizeof(int32_t));
  auto* input_keys = static_cast<uint64_t*>(keys_in.data);
  auto* output_keys = static_cast<uint64_t*>(keys_out.data);
  auto* input_ids = static_cast<int32_t*>(ids_in.data);
  auto* output_ids = static_cast<int32_t*>(ids_out.data);
  size_t bytes = 0;
  if (is_standard_integer(DType(p.input_dtype(KernelInput::A))) || p.input_dtype(KernelInput::A) == int(DType::Float64)) {
    const int groups = count / length;
    SortBuffer segment_offsets((size_t(groups) + 1) * sizeof(int32_t));
    auto* offsets = static_cast<int32_t*>(segment_offsets.data);
    make_wide_sort_keys<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(input_keys, input_ids, offsets, a, p, length, inner);
    check_sort(cudaGetLastError());
    // Full-width keys need independent segments rather than packed row bits.
    // Stable radix sorting retains original indices for duplicate values.
    check_sort(cub::DeviceSegmentedRadixSort::SortPairs(nullptr, bytes, input_keys, output_keys, input_ids, output_ids,
                count, groups, offsets, offsets + 1, 0, 64, current_cuda_stream()));
    SortBuffer scratch(bytes);
    check_sort(cub::DeviceSegmentedRadixSort::SortPairs(scratch.data, bytes, input_keys, output_keys, input_ids, output_ids,
                count, groups, offsets, offsets + 1, 0, 64, current_cuda_stream()));
  } else {
    make_sort_keys<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(input_keys, input_ids, a, p, length, inner);
    check_sort(cudaGetLastError());
    check_sort(cub::DeviceRadixSort::SortPairs(nullptr, bytes, input_keys, output_keys, input_ids, output_ids, count, 0, 64, current_cuda_stream()));
    SortBuffer scratch(bytes);
    // Floating-point keys preserve the established signed-zero and NaN order.
    check_sort(cub::DeviceRadixSort::SortPairs(scratch.data, bytes, input_keys, output_keys, input_ids, output_ids, count, 0, 64, current_cuda_stream()));
  }
  store_sort_indices<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(out, output_ids, count, length, inner);
  check_sort(cudaGetLastError());
}
} // namespace elman
