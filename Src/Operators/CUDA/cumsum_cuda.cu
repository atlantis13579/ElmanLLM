#include "kernel.cuh"
#include "Core/AllocationPool.h"
#include "Operators/scan_params.h"
#include "Operators/integer_reduction.h"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace elman {
namespace {
void check_scan(cudaError_t error) {
  if (error != cudaSuccess) {
    cudaGetLastError();
    throw std::runtime_error(std::string("CUDA cumsum: ") + cudaGetErrorString(error));
  }
}
struct ScanBuffer {
  BufferPtr storage;
  void* data;
  explicit ScanBuffer(size_t bytes) : storage(backend_workspace(Backend::CUDA, bytes)), data(storage->raw_pointer()) {}
  ScanBuffer(const ScanBuffer&) = delete;
  ScanBuffer& operator=(const ScanBuffer&) = delete;
};

template <class T> __device__ T scan_add(T a, T b) { return T(a + b); }

template <class T> __device__ T scan_thread_total(const T (&values)[16]) {
  // Keep the sequential per-thread order used by the CUDA 12.8 reference.
  // Newer CUB releases may select a tree or packed half reduction instead.
  T total = values[0];
  #pragma unroll
  for (int i = 1; i < 16; ++i) total = scan_add(total, values[i]);
  return total;
}

template <class T> __global__ void reverse_scan_input(T* output, const T* input, int count) {
  int i = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (i < count) output[i] = input[count - 1 - i];
}

template <class T> __global__ void scan_strided(T* output, const T* input, int count,
                                               int length, int inner, bool reverse) {
  int group = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (group >= count / length) return;
  int base = group / inner * length * inner + group % inner;
  T value = T(0.0f);
  for (int j = 0; j < length; ++j) {
    int i = base + (reverse ? length - 1 - j : j) * inner;
    value = scan_add(value, input[i]);
    output[i] = value;
  }
}

template <class T> __global__ void scan_rows(T* output, const T* input, int rows, int length,
                                            int logarithm, bool reverse) {
  extern __shared__ uint32_t storage[];
  int width = int(blockDim.x), lane = int(threadIdx.x);
  int row = int(blockIdx.x * blockDim.y + threadIdx.y);
  T* values = reinterpret_cast<T*>(storage) + threadIdx.y * width * 2;
  T carry = T(0.0f);
  for (int start = 0; start < length; start += width * 2) {
    for (int half = 0; half < 2; ++half) {
      int column = start + half * width + lane;
      values[half * width + lane] = row < rows && column < length
          ? input[row * length + (reverse ? length - 1 - column : column)] : T(0.0f);
    }
    if (lane == 0) values[0] = scan_add(values[0], carry);
    __syncthreads();
    // Sklansky inclusive scan over two elements per lane.
    for (int stage = 0; stage <= logarithm; ++stage) {
      int span = 1 << stage;
      int right = (lane / span) * span * 2 + span;
      int target = right + lane % span;
      values[target] = scan_add(values[target], values[right - 1]);
      __syncthreads();
    }
    for (int half = 0; half < 2; ++half) {
      int column = start + half * width + lane;
      if (row < rows && column < length)
        output[row * length + (reverse ? length - 1 - column : column)] = values[half * width + lane];
    }
    carry = values[width * 2 - 1];
    __syncthreads();
  }
}

// The deterministic vector path assigns a fixed range to each block and
// computes block prefixes in a separate pass; execution order cannot affect it.
template <class T> __global__ void scan_block_totals(const T* input, T* totals, int count, int iterations) {
  constexpr int Threads = sizeof(T) >= 8 ? 256 : 512;
  constexpr int Tile = Threads * 16;
  using Load = cub::BlockLoad<T, Threads, 16, cub::BLOCK_LOAD_STRIPED>;
  using Reduce = cub::BlockReduce<T, Threads>;
  __shared__ union { typename Load::TempStorage load; typename Reduce::TempStorage reduce; } shared;
  int64_t start = int64_t(blockIdx.x) * iterations * Tile;
  T total = T(0.0f), values[16];
  for (int iteration = 0; iteration < iterations && start < count; ++iteration, start += Tile) {
    int valid = int(min(int64_t(Tile), int64_t(count) - start));
    Load(shared.load).Load(input + start, values, valid, T(0.0f));
    __syncthreads();
    total = scan_add(total, Reduce(shared.reduce).Sum(scan_thread_total(values)));
    __syncthreads();
  }
  if (threadIdx.x == 0) totals[blockIdx.x] = total;
}

template <class T> struct ScanPrefix {
  T total;
  __device__ T operator()(T value) {
    T previous = total;
    total = scan_add(total, value);
    return previous;
  }
};

template <class T> __global__ void scan_vector(const T* input, T* output, const T* totals, int count, int iterations) {
  constexpr int Threads = sizeof(T) >= 8 ? 256 : 512;
  constexpr int Tile = Threads * 16;
  using Load = cub::BlockLoad<T, Threads, 16, cub::BLOCK_LOAD_WARP_TRANSPOSE>;
  using Store = cub::BlockStore<T, Threads, 16, cub::BLOCK_STORE_WARP_TRANSPOSE>;
  using Scan = cub::BlockScan<T, Threads, cub::BLOCK_SCAN_WARP_SCANS>;
  using Reduce = cub::BlockReduce<T, Threads>;
  __shared__ union {
    typename Load::TempStorage load; typename Store::TempStorage store;
    typename Scan::TempStorage scan; typename Reduce::TempStorage reduce;
  } shared;
  T prefix = threadIdx.x < blockIdx.x ? totals[threadIdx.x] : T(0.0f);
  if (threadIdx.x + Threads < blockIdx.x) prefix = scan_add(prefix, totals[threadIdx.x + Threads]);
  prefix = Reduce(shared.reduce).Sum(prefix);
  __syncthreads();
  ScanPrefix<T> callback{prefix};
  T values[16];
  int64_t start = int64_t(blockIdx.x) * iterations * Tile;
  for (int iteration = 0; iteration < iterations && start < count; ++iteration, start += Tile) {
    int valid = int(min(int64_t(Tile), int64_t(count) - start));
    Load(shared.load).Load(input + start, values, valid, T(0.0f));
    __syncthreads();
    T thread_prefix = scan_thread_total(values);
    Scan(shared.scan).ExclusiveSum(thread_prefix, thread_prefix, callback);
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
      thread_prefix = scan_add(thread_prefix, values[i]);
      values[i] = thread_prefix;
    }
    __syncthreads();
    Store(shared.store).Store(output + start, values, valid);
    __syncthreads();
  }
}

__global__ void scan_integers(uint64_t* output, const uint32_t* a, KernelParams p,
                             int length, int inner) {
  unsigned int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= unsigned(p.output_count() / length)) return;
  int base = int(group) / inner * length * inner + int(group) % inner;
  uint64_t value = 0;
  for (int j = 0; j < length; ++j) {
    int i = base + j * inner;
    value += integer_reduction_value(a, i, p.input_dtype(KernelInput::A));
    output[i] = value;
  }
}

template <class T> void launch_float_scan(T* output, const T* input, int count, int length, int inner, bool reverse) {
  constexpr int Threads = sizeof(T) >= 8 ? 256 : 512;
  constexpr int Tile = Threads * 16;
  if (count == length) {
    int device = 0, multiprocessors = 0;
    check_scan(cudaGetDevice(&device));
    check_scan(cudaDeviceGetAttribute(&multiprocessors, cudaDevAttrMultiProcessorCount, device));
    int blocks = int((int64_t(count) + (Tile - 1)) / Tile);
    int iterations = (blocks + multiprocessors - 1) / multiprocessors;
    blocks = std::min(blocks, multiprocessors);
    if (blocks > Threads * 2) throw std::runtime_error("Too many CUDA blocks for deterministic cumsum");
    ScanBuffer totals(size_t(blocks) * sizeof(T));
    if (reverse) {
      ScanBuffer reversed(size_t(count) * sizeof(T)), scanned(size_t(count) * sizeof(T));
      reverse_scan_input<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(static_cast<T*>(reversed.data), input, count);
      scan_block_totals<<<blocks, Threads, 0, current_cuda_stream()>>>(static_cast<T*>(reversed.data), static_cast<T*>(totals.data), count, iterations);
      scan_vector<<<blocks, Threads, 0, current_cuda_stream()>>>(static_cast<T*>(reversed.data), static_cast<T*>(scanned.data), static_cast<T*>(totals.data), count, iterations);
      reverse_scan_input<<<(unsigned(count) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(output, static_cast<T*>(scanned.data), count);
    } else {
      scan_block_totals<<<blocks, Threads, 0, current_cuda_stream()>>>(input, static_cast<T*>(totals.data), count, iterations);
      scan_vector<<<blocks, Threads, 0, current_cuda_stream()>>>(input, output, static_cast<T*>(totals.data), count, iterations);
    }
    check_scan(cudaGetLastError());
  } else if (inner == 1) {
    int rows = count / length, row_log = 0, length_log = 0;
    while ((int64_t(1) << row_log) < rows) ++row_log;
    while ((int64_t(1) << length_log) < length) ++length_log;
    uint32_t balance = (9u + uint32_t(length_log) - uint32_t(row_log)) / 2u;
    int logarithm = int(std::max(4u, std::min(9u, balance)));
    int width = 1 << logarithm, block_rows = 512 / width;
    scan_rows<<<(unsigned(rows) + block_rows - 1) / block_rows, dim3(width, block_rows), 1024 * sizeof(T), current_cuda_stream()>>>(
        output, input, rows, length, logarithm, reverse);
  } else {
    int groups = count / length;
    scan_strided<<<(unsigned(groups) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(output, input, count, length, inner, reverse);
  }
}

void dispatch_scan(uint32_t* out, const uint32_t* a, const KernelParams& p, bool reverse) {
  if (!p.output_count()) return;
  int axis = p.attribute(operator_attr::scan::kAxis);
  bool scalar = p.input_rank(KernelInput::A) == 0;
  int length = scalar ? 1 : p.input_shape(KernelInput::A, axis);
  int inner = scalar ? 1 : p.input_stride(KernelInput::A, axis);
  if (p.output_dtype() == int(DType::Int64)) {
    int groups = p.output_count() / length;
    scan_integers<<<(unsigned(groups) + 255u) / 256u, 256, 0, current_cuda_stream()>>>(reinterpret_cast<uint64_t*>(out), a, p, length, inner);
    return;
  }
  if (scalar || (reverse && length == 1)) {
    size_t bytes = size_t(p.output_count()) * (p.output_dtype() == int(DType::Float64) ? 8 : p.output_dtype() == int(DType::Float32) ? 4 : 2);
    check_scan(cudaMemcpyAsync(out, a, bytes, cudaMemcpyDeviceToDevice, current_cuda_stream()));
    return;
  }
  if (p.output_dtype() == int(DType::Float64))
    launch_float_scan(reinterpret_cast<double*>(out), reinterpret_cast<const double*>(a), p.output_count(), length, inner, reverse);
  else if (p.output_dtype() == int(DType::Float32))
    launch_float_scan(reinterpret_cast<float*>(out), reinterpret_cast<const float*>(a), p.output_count(), length, inner, reverse);
  else if (p.output_dtype() == int(DType::Float16))
    launch_float_scan(reinterpret_cast<__half*>(out), reinterpret_cast<const __half*>(a), p.output_count(), length, inner, reverse);
  else
    launch_float_scan(reinterpret_cast<__nv_bfloat16*>(out), reinterpret_cast<const __nv_bfloat16*>(a), p.output_count(), length, inner, reverse);
}
}
void launch_cumsum_cuda(uint32_t* out, const uint32_t* a, const uint32_t*, const uint32_t*, const KernelParams& p) {
  dispatch_scan(out, a, p, false);
}
void launch_cumsum_backward_cuda(uint32_t* out, const uint32_t* a, const uint32_t*, const uint32_t*, const KernelParams& p) {
  dispatch_scan(out, a, p, true);
}
} // namespace elman
