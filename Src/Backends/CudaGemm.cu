#include "Backends/CudaGemm.h"
#include "Backends/AttentionGemm.h"
#include "Backends/CudaConvolution.h"
#include "Backends/CudaExecution.h"
#include "Kernals/CUDA/kernel.cuh"
#include "Kernals/matmul_params.h"
#include <mma.h>
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <type_traits>

namespace elman {
namespace {
// Native GEMM: out = alpha * op(A)[M,K] * op(B)[K,N] + beta * bias.
// TA/TB describe stored transpose views; no temporary transpose is allocated.
// FP32 uses CUDA-core FMA without narrowing inputs to TF32/FP16. FP16/BF16 use
// WMMA with FP32 accumulation where supported, then convert only at the store.
//
// Design sources (referenced by number beside each algorithm):
// [1] CUTLASS Efficient GEMM: hierarchical tiles, epilogues and parallel K reduction.
//     https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html
// [2] CUTLASS MmaPipelined: two shared stages and register-fragment pipelining.
//     https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_pipelined.h
// [3] Triton matmul tutorial: grouped tile IDs for L2 reuse.
//     https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html#l2-cache-optimizations
// [4] NVIDIA CUDA Best Practices: coalescing, shared-memory banks and padding.
//     https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#shared-memory-and-memory-banks
// [5] NVIDIA CUDA Programming Guide: WMMA layouts, alignment and warp participation.
//     https://docs.nvidia.com/cuda/archive/12.6.3/cuda-c-programming-guide/index.html#warp-matrix-functions
// [6] NVIDIA warp primitives tutorial: shuffle reductions and explicit synchronization.
//     https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
//
// These are design references, not runtime dependencies. The kernels implement
// Elman's layouts, batching and epilogues directly. Tile sizes, lane maps and
// dispatch thresholds below are local choices measured on an RTX 4060 (SM89),
// not CUTLASS/Triton defaults or claims of optimal tuning on every GPU.
struct Matrix {
  int rows = 0, columns = 0, shared = 0, lda = 0, ldb = 0, ldc = 0;
  int batches = 1, groups = 1, batch_base = 0;
  int64_t stride_a = 0, stride_b = 0, stride_out = 0;
  int batch_rank = 0, shape[6]{}, a_step[6]{}, b_step[6]{};
  // Device pointer batches use the attention ABI: B[], A[], output[].
  void* const* pointers = nullptr;
  const void* bias = nullptr;
  int bias_row = 0, bias_column = 0;
  float alpha = 1, beta = 0;
  bool output_float = false;
};
thread_local bool convolution_tf32 = false;

template<class T> __device__ float read(T value) { return float(value); }
template<> __device__ float read(__half value) { return __half2float(value); }
template<> __device__ float read(__nv_bfloat16 value) { return __bfloat162float(value); }
template<class T> __device__ T convert(float value) { return T(value); }
template<> __device__ __half convert(float value) { return __float2half_rn(value); }
template<> __device__ __nv_bfloat16 convert(float value) { return __float2bfloat16_rn(value); }

template<class T>
__device__ void batch_pointers(const Matrix& s, int batch, const T*& a, const T*& b, void*& out) {
  batch += s.batch_base;
  if (s.pointers) {
    b = static_cast<const T*>(s.pointers[batch]);
    a = static_cast<const T*>(s.pointers[s.batches + batch]);
    out = s.pointers[2 * size_t(s.batches) + batch];
    return;
  }
  int64_t ao = int64_t(batch) * s.stride_a, bo = int64_t(batch / s.groups) * s.stride_b;
  if (s.batch_rank) {
    ao = bo = 0;
    int coordinate = batch;
    for (int axis = s.batch_rank - 1; axis >= 0; --axis) {
      const int index = coordinate % s.shape[axis];
      coordinate /= s.shape[axis];
      ao += int64_t(index) * s.a_step[axis];
      bo += int64_t(index) * s.b_step[axis];
    }
  }
  // Empty products can have null input allocations.
  if (s.shared) { a += ao; b += bo; }
  out = static_cast<char*>(out) + int64_t(batch) * s.stride_out * (s.output_float ? sizeof(float) : sizeof(T));
}

// Fuse scaling, broadcast bias and conversion into the final write, following
// the epilogue principle in [1]. Zero bias strides implement broadcasting;
// beta == 0 skips the read so an ignored NaN addend cannot contaminate the sum.
template<class T>
__device__ void store_result(void* out, const Matrix& s, int row, int column, float sum) {
  float value = s.alpha * sum;
  if (s.beta != 0)
    value += s.beta * read(static_cast<const T*>(s.bias)[size_t(row) * s.bias_row + size_t(column) * s.bias_column]);
  const size_t index = size_t(row) * s.ldc + column;
  if (s.output_float) static_cast<float*>(out)[index] = value;
  else static_cast<T*>(out)[index] = convert<T>(value);
}

// Adapt [3]'s grouped program order: neighboring block IDs traverse up to four
// M tiles before advancing N, encouraging reuse of the same B panel in L2.
// The final group can be shorter; divide by its actual size to avoid holes.
// This is a locality hint, not a dependency on CUDA block execution order.
template<int BM, int BN>
__device__ void tile_origin(const Matrix& s, int& row, int& column) {
  const int mr = (s.rows + BM - 1) / BM, nc = (s.columns + BN - 1) / BN;
  const int first = int(blockIdx.x) / (4 * nc) * 4;
  const int group = min(4, mr - first), within = int(blockIdx.x) % (4 * nc);
  row = (first + within % group) * BM;
  column = (within / group) * BN;
}

// Apply [4]'s contiguous-access principle in the stored orientation: read four
// adjacent elements per packet. Alignment and full-packet bounds are checked
// before a vector read; scalar zero-filled tails also support sliced tensors.
template<class T>
__device__ float4 load_four(const T* source, int row, int column, int rows, int columns, int leading) {
  if constexpr (std::is_same_v<T, float>) {
    if (row < rows && column + 3 < columns &&
        (reinterpret_cast<uintptr_t>(source + size_t(row) * leading + column) & 15) == 0)
      return *reinterpret_cast<const float4*>(source + size_t(row) * leading + column);
  } else {
    if (row < rows && column + 3 < columns &&
        (reinterpret_cast<uintptr_t>(source + size_t(row) * leading + column) & 7) == 0) {
      const uint2 packet = *reinterpret_cast<const uint2*>(source + size_t(row) * leading + column);
      if constexpr (std::is_same_v<T, __half>)
        return make_float4(__half2float(__ushort_as_half(uint16_t(packet.x))), __half2float(__ushort_as_half(uint16_t(packet.x >> 16))),
                           __half2float(__ushort_as_half(uint16_t(packet.y))), __half2float(__ushort_as_half(uint16_t(packet.y >> 16))));
      else
        return make_float4(__bfloat162float(__ushort_as_bfloat16(uint16_t(packet.x))), __bfloat162float(__ushort_as_bfloat16(uint16_t(packet.x >> 16))),
                           __bfloat162float(__ushort_as_bfloat16(uint16_t(packet.y))), __bfloat162float(__ushort_as_bfloat16(uint16_t(packet.y >> 16))));
    }
  }
  float values[4];
  #pragma unroll
  for (int i = 0; i < 4; ++i)
    values[i] = row < rows && column + i < columns ? read(source[size_t(row) * leading + column + i]) : 0.f;
  return make_float4(values[0], values[1], values[2], values[3]);
}

// Baseline SIMT tiling, based on [1]'s threadblock/thread hierarchy. A block
// stages BM*BK + BK*BN values and reuses them for BM*BN*BK multiply-adds.
// Each thread keeps a TM*TN outer product (usually 4x4) in registers, reducing
// shared reads per FMA. TA/TB preserve contiguous global loads; shared storage
// normalizes both operands to [K][M/N]. The extra column reduces bank aliasing
// during transpose staging ([4]). Two barriers protect load/use and use/reuse.
template<class T, bool TA, bool TB, int BM, int BN = 64, int TM = 4, int TN = 4, int BK = 32>
__global__ void simt_gemm(void* out, const T* a, const T* b, Matrix s) {
  constexpr int threads = BM * BN / (TM * TN);
  __shared__ float left[BK][BM + 1], right[BK][BN + 1];
  batch_pointers(s, int(blockIdx.y), a, b, out);
  int row, column;
  tile_origin<BM, BN>(s, row, column);
  const int tid = int(threadIdx.x), tr = tid / (BN / TN), tc = tid % (BN / TN);
  float accum[TM][TN]{};
  for (int start = 0; start < s.shared; start += BK) {
    #pragma unroll
    for (int load = 0; load < (BM * BK + 4 * threads - 1) / (4 * threads); ++load) {
      const int i = (tid + load * threads) * 4;
      if (i < BM * BK) {
        const int r = TA ? i % BM : i / BK, k = TA ? i / BM : i % BK;
        const float4 packet = load_four(a, TA ? start + k : row + r, TA ? row + r : start + k,
                                        TA ? s.shared : s.rows, TA ? s.rows : s.shared, s.lda);
        const float values[4]{packet.x, packet.y, packet.z, packet.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) left[k + (TA ? 0 : j)][r + (TA ? j : 0)] = values[j];
      }
    }
    #pragma unroll
    for (int load = 0; load < (BN * BK + 4 * threads - 1) / (4 * threads); ++load) {
      const int i = (tid + load * threads) * 4;
      if (i < BN * BK) {
        const int n = TB ? i / BK : i % BN, k = TB ? i % BK : i / BN;
        const float4 packet = load_four(b, TB ? column + n : start + k, TB ? start + k : column + n,
                                        TB ? s.columns : s.shared, TB ? s.shared : s.columns, s.ldb);
        const float values[4]{packet.x, packet.y, packet.z, packet.w};
        #pragma unroll
        for (int j = 0; j < 4; ++j) right[k + (TB ? j : 0)][n + (TB ? 0 : j)] = values[j];
      }
    }
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      float av[TM], bv[TN];
      #pragma unroll
      for (int i = 0; i < TM; ++i) av[i] = left[k][tr + i * (BM / TM)];
      #pragma unroll
      for (int j = 0; j < TN; ++j) bv[j] = right[k][tc + j * (BN / TN)];
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) accum[i][j] = fmaf(av[i], bv[j], accum[i][j]);
    }
    __syncthreads();
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int r = row + tr + i * (BM / TM), n = column + tc + j * (BN / TN);
      if (r < s.rows && n < s.columns) store_result<T>(out, s, r, n, accum[i][j]);
    }
}

// Global-to-register stage of [2]'s software pipeline. Load the next K tile
// before consuming the current tile, making independent FMAs available to hide
// memory latency. These are ordinary loads, not cp.async or a TF32 conversion.
// Flatten each physically stored tile so every packet follows its contiguous
// dimension; stage_fp32 performs any layout conversion in shared memory.
template<bool TA, bool TB, int BM, int BN, int BK, int Threads>
__device__ void prefetch_fp32(float4 (&av)[BM * BK / (4 * Threads)], float4 (&bv)[BN * BK / (4 * Threads)],
                              const float* a, const float* b, const Matrix& s, int row, int column, int start) {
  const int tid = int(threadIdx.x);
  #pragma unroll
  for (int i = 0; i < BM * BK / (4 * Threads); ++i) {
    const int flat = (tid + i * Threads) * 4;
    const int r = TA ? flat % BM : flat / BK, k = TA ? flat / BM : flat % BK;
    av[i] = load_four(a, TA ? start + k : row + r, TA ? row + r : start + k,
                      TA ? s.shared : s.rows, TA ? s.rows : s.shared, s.lda);
  }
  #pragma unroll
  for (int i = 0; i < BN * BK / (4 * Threads); ++i) {
    const int flat = (tid + i * Threads) * 4;
    const int n = TB ? flat / BK : flat % BN, k = TB ? flat % BK : flat / BN;
    bv[i] = load_four(b, TB ? column + n : start + k, TB ? start + k : column + n,
                      TB ? s.columns : s.shared, TB ? s.shared : s.columns, s.ldb);
  }
}

// Convert prefetched packets into left[K][M] and right[K][N]. For a physically
// contiguous M/N dimension, use an aligned float4 store; otherwise scatter the
// four K values. Both layouts then share one register-FMA mainloop. Unlike the
// scalar SIMT tile, these pitches retain 16-byte alignment for fragment loads.
template<bool TA, bool TB, int BM, int BN, int BK, int Threads>
__device__ void stage_fp32(float* left, float* right,
                           const float4 (&av)[BM * BK / (4 * Threads)], const float4 (&bv)[BN * BK / (4 * Threads)]) {
  const int tid = int(threadIdx.x);
  #pragma unroll
  for (int i = 0; i < BM * BK / (4 * Threads); ++i) {
    const int flat = (tid + i * Threads) * 4;
    const int r = TA ? flat % BM : flat / BK, k = TA ? flat / BM : flat % BK;
    if constexpr (TA) *reinterpret_cast<float4*>(left + k * BM + r) = av[i];
    else {
      left[(k + 0) * BM + r] = av[i].x; left[(k + 1) * BM + r] = av[i].y;
      left[(k + 2) * BM + r] = av[i].z; left[(k + 3) * BM + r] = av[i].w;
    }
  }
  #pragma unroll
  for (int i = 0; i < BN * BK / (4 * Threads); ++i) {
    const int flat = (tid + i * Threads) * 4;
    const int n = TB ? flat / BK : flat % BN, k = TB ? flat % BK : flat / BN;
    if constexpr (!TB) *reinterpret_cast<float4*>(right + k * BN + n) = bv[i];
    else {
      right[(k + 0) * BN + n] = bv[i].x; right[(k + 1) * BN + n] = bv[i].y;
      right[(k + 2) * BN + n] = bv[i].z; right[(k + 3) * BN + n] = bv[i].w;
    }
  }
}

// Local lane layout derived using [4]'s 32-bank model. Eight lanes read one
// 32-column strip as float4 packets; other row groups reuse those B addresses.
// TN=8 reads a second strip at +32, instead of spacing adjacent lanes by eight
// floats (which aliases banks within the eight-lane group). The epilogue must
// use the same strip mapping; adjacent register entries need not be adjacent C.
template<int TM, int TN>
__device__ void fragment_fp32(float (&av)[TM], float (&bv)[TN], const float* a, const float* b) {
  #pragma unroll
  for (int i = 0; i < TM; i += 4) {
    const float4 value = *reinterpret_cast<const float4*>(a + i);
    av[i] = value.x; av[i + 1] = value.y; av[i + 2] = value.z; av[i + 3] = value.w;
  }
  #pragma unroll
  for (int i = 0; i < TN; i += 4) {
    const float4 value = *reinterpret_cast<const float4*>(b + (i / 4) * 32);
    bv[i] = value.x; bv[i + 1] = value.y; bv[i + 2] = value.z; bv[i + 3] = value.w;
  }
}

// FP32 warp tiling combines [1]'s register reuse with [2]'s two-level pipeline:
//   block 64x64  -> four warps, 32x32 per warp, 8x4 outputs per thread;
//   block 128x128 -> eight warps, 32x64 per warp, 8x8 outputs per thread.
// A warp has four row groups of eight lanes. Each group shares A values while
// its lanes cover B's 32-column strips. Larger tiles reuse loads more but cost
// more registers and waste more work at edges; launch() chooses between them.
// BK=8 bounds prefetch/register pressure. Two shared stages occupy 8/16 KiB;
// av/bv also alternate between two register fragments to overlap shared loads
// with the current outer product. All FMAs and accumulators remain full FP32.
template<bool TA, bool TB, int BM = 64, int BN = 64, int TN = 4>
__global__ void pipelined_fp32_gemm(void* out, const float* a, const float* b, Matrix s) {
  constexpr int BK = 8, TM = 8, WM = TM * 4, WN = TN * 8;
  constexpr int threads = (BM / WM) * (BN / WN) * 32;
  static_assert(BM * BK % (4 * threads) == 0 && BN * BK % (4 * threads) == 0);
  __shared__ __align__(16) float left[2][BK * BM], right[2][BK * BN];
  batch_pointers(s, int(blockIdx.y), a, b, out);
  int row, column;
  tile_origin<BM, BN>(s, row, column);
  const int warp = int(threadIdx.x) / 32, lane = int(threadIdx.x) % 32;
  const int r = warp / (BN / WN) * WM + lane / 8 * TM;
  const int n = warp % (BN / WN) * WN + lane % 8 * 4;
  float accum[TM][TN]{};
  float4 next_a[BM * BK / (4 * threads)], next_b[BN * BK / (4 * threads)];
  // Prologue: make stage zero visible to every warp before its first read.
  prefetch_fp32<TA, TB, BM, BN, BK, threads>(next_a, next_b, a, b, s, row, column, 0);
  stage_fp32<TA, TB, BM, BN, BK, threads>(left[0], right[0], next_a, next_b);
  __syncthreads();
  int stage = 0;
  for (int start = 0; start < s.shared; start += BK) {
    const bool more = start + BK < s.shared;
    if (more) prefetch_fp32<TA, TB, BM, BN, BK, threads>(next_a, next_b, a, b, s, row, column, start + BK);
    float av[2][TM], bv[2][TN];
    fragment_fp32(av[0], bv[0], left[stage] + r, right[stage] + n);
    #pragma unroll
    for (int k = 0; k < BK; ++k) {
      // Fetch k+1 into the other register fragment before computing k; this
      // exposes independent shared-load/FMA instructions to the scheduler.
      if (k + 1 < BK)
        fragment_fp32(av[(k + 1) % 2], bv[(k + 1) % 2], left[stage] + (k + 1) * BM + r, right[stage] + (k + 1) * BN + n);
      #pragma unroll
      for (int i = 0; i < TM; ++i)
        #pragma unroll
        for (int j = 0; j < TN; ++j) accum[i][j] = fmaf(av[k % 2][i], bv[k % 2][j], accum[i][j]);
    }
    if (more) {
      // Only overwrite the alternate stage. The previous iteration's barrier
      // guarantees nobody still reads it; this barrier publishes the new tile
      // and finishes all reads of the current stage before the next overwrite.
      // All threads take this branch together, including masked tail threads.
      stage_fp32<TA, TB, BM, BN, BK, threads>(left[stage ^ 1], right[stage ^ 1], next_a, next_b);
      __syncthreads();
      stage ^= 1;
    }
  }
  #pragma unroll
  for (int i = 0; i < TM; ++i)
    #pragma unroll
    for (int j = 0; j < TN; ++j) {
      const int c = column + n + (j / 4) * 32 + j % 4;
      if (row + r + i < s.rows && c < s.columns)
        store_result<float>(out, s, row + r + i, c, accum[i][j]);
    }
}

// Decode/GEMV specialization using [6]'s shuffle reduction. With few output
// rows, large GEMM tiles do little useful work. NT assigns one weight row to
// each warp: lanes read four adjacent K values, then a shuffle tree sums them.
// NN instead assigns columns to lanes and K residues to eight warps, preserving
// contiguous B reads; shared partial sums merge the warps. The output ownership
// and dispatch are local choices, not a copied library GEMV implementation.
template<class T, bool TB>
__global__ void small_gemm(void* out, const T* a, const T* b, Matrix s) {
  batch_pointers(s, int(blockIdx.y), a, b, out);
  const int lane = int(threadIdx.x) % 32, warp = int(threadIdx.x) / 32;
  const int tiles = (s.columns + (TB ? 7 : 31)) / (TB ? 8 : 32);
  const int row = int(blockIdx.x) / tiles;
  const int column = int(blockIdx.x) % tiles * (TB ? 8 : 32) + (TB ? warp : lane);
  float sum = 0;
  if (column < s.columns) {
    if constexpr (TB) {
      for (int k = lane * 4; k < s.shared; k += 128) {
        const float4 av = load_four(a, row, k, s.rows, s.shared, s.lda);
        const float4 bv = load_four(b, column, k, s.columns, s.shared, s.ldb);
        sum = fmaf(av.x, bv.x, sum); sum = fmaf(av.y, bv.y, sum);
        sum = fmaf(av.z, bv.z, sum); sum = fmaf(av.w, bv.w, sum);
      }
    } else {
      for (int k = warp; k < s.shared; k += 8)
        sum = fmaf(read(a[size_t(row) * s.lda + k]), read(b[size_t(k) * s.ldb + column]), sum);
    }
  }
  if constexpr (TB) {
    // Invalid columns still participate with zero: the full warp mask is valid.
    #pragma unroll
    for (int offset = 16; offset; offset /= 2) sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (!lane && column < s.columns) store_result<T>(out, s, row, column, sum);
  } else {
    __shared__ float partial[8][32];
    partial[warp][lane] = sum;
    __syncthreads();
    if (!warp && column < s.columns) {
      #pragma unroll
      for (int i = 1; i < 8; ++i) sum += partial[i][lane];
      store_result<T>(out, s, row, column, sum);
    }
  }
}

// Vectorized NN decode variant of the same [4]/[6] access/reduction principles.
// A block computes one row x 32 columns. Each eight-lane group shares one A
// element and loads 32 contiguous B values as eight float4 packets. There are
// 32 such groups partitioning K. Shuffle offsets 16 and 8 combine the four
// groups within each warp; shared memory then combines the eight warp results.
template<class T>
__global__ void small_nn_gemm(void* out, const T* a, const T* b, Matrix s) {
  __shared__ float partial[8][32];
  batch_pointers(s, int(blockIdx.y), a, b, out);
  const int tid = int(threadIdx.x), lane = tid % 32, warp = tid / 32;
  const int group = tid / 8, column_group = tid % 8;
  const int tiles = (s.columns + 31) / 32, row = int(blockIdx.x) / tiles;
  const int column = int(blockIdx.x) % tiles * 32 + column_group * 4;
  float sums[4]{};
  for (int k = group; k < s.shared; k += 32) {
    const float av = read(a[size_t(row) * s.lda + k]);
    const float4 bv = load_four(b, k, column, s.shared, s.columns, s.ldb);
    sums[0] = fmaf(av, bv.x, sums[0]); sums[1] = fmaf(av, bv.y, sums[1]);
    sums[2] = fmaf(av, bv.z, sums[2]); sums[3] = fmaf(av, bv.w, sums[3]);
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    sums[j] += __shfl_down_sync(0xffffffff, sums[j], 16);
    sums[j] += __shfl_down_sync(0xffffffff, sums[j], 8);
    if (lane < 8) partial[warp][column_group * 4 + j] = sums[j];
  }
  __syncthreads();
  if (!warp) {
    const int n = int(blockIdx.x) % tiles * 32 + lane;
    if (n < s.columns) {
      float value = partial[0][lane];
      #pragma unroll
      for (int i = 1; i < 8; ++i) value += partial[i][lane];
      store_result<T>(out, s, row, n, value);
    }
  }
}

// Apply [4]'s vector loads and [5]'s WMMA layout/alignment requirements: copy
// eight FP16/BF16 elements per aligned 128-bit packet in the stored orientation.
// WMMA row/column-major fragments interpret that tile without a transpose.
// Callers pad the pitch by eight 16-bit elements, retaining WMMA's 16-byte
// leading-dimension granularity while changing the shared bank mapping.
// Scalar zero-filled tails handle odd pitches and unaligned global slices;
// shared tile bases remain 32-byte aligned for WMMA loads.
template<class T, int Rows, int Columns, int Pitch, int Threads>
__device__ void stage_tensor(T* tile, const T* source, int leading, int row, int column,
                             int rows, int columns, int thread) {
  #pragma unroll
  for (int load = 0; load < Rows * Columns / (8 * Threads); ++load) {
    const int index = (thread + load * Threads) * 8;
    const int r = index / Columns, c = index % Columns;
    if (row + r < rows && column + c + 7 < columns &&
        (reinterpret_cast<uintptr_t>(source + size_t(row + r) * leading + column + c) & 15) == 0) {
      *reinterpret_cast<uint4*>(tile + r * Pitch + c) =
          *reinterpret_cast<const uint4*>(source + size_t(row + r) * leading + column + c);
    } else {
      #pragma unroll
      for (int j = 0; j < 8; ++j)
        tile[r * Pitch + c + j] = row + r < rows && column + c + j < columns
            ? source[size_t(row + r) * leading + column + c + j] : convert<T>(0);
    }
  }
}

// Tensor-core tiling uses [1]'s hierarchy and [5]'s collective WMMA operations.
// Four warps cover a 32x64 block; each owns two 16x16 output fragments. For each
// K=32 stage, every warp performs two K=16 MMA steps. Operands stay FP16/BF16
// and accumulators stay FP32. All lanes execute each WMMA call, even at tails.
// Fragment element ownership is opaque, so first store a dense shared result,
// synchronize, then apply bounds/bias/scaling with ordinary row/column indices.
template<class T, bool TA, bool TB>
__global__ void tensor_gemm(void* out, const T* a, const T* b, Matrix s) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  constexpr int BM = 32, BN = 64, BK = 32;
  constexpr int apitch = (TA ? BM : BK) + 8, bpitch = (TB ? BK : BN) + 8;
  __shared__ __align__(32) T left[(TA ? BK : BM) * apitch], right[(TB ? BN : BK) * bpitch];
  __shared__ __align__(32) float result[BM * BN];
  batch_pointers(s, int(blockIdx.y), a, b, out);
  int row, column;
  tile_origin<BM, BN>(s, row, column);
  const int tid = int(threadIdx.x), warp = tid / 32, wr = (warp / 2) * 16, wn = (warp % 2) * 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2];
  wmma::fill_fragment(accum[0], 0.f);
  wmma::fill_fragment(accum[1], 0.f);
  for (int start = 0; start < s.shared; start += BK) {
    stage_tensor<T, TA ? BK : BM, TA ? BM : BK, apitch, 128>(left, a, s.lda,
        TA ? start : row, TA ? row : start, TA ? s.shared : s.rows, TA ? s.rows : s.shared, tid);
    stage_tensor<T, TB ? BN : BK, TB ? BK : BN, bpitch, 128>(right, b, s.ldb,
        TB ? column : start, TB ? start : column, TB ? s.columns : s.shared, TB ? s.shared : s.columns, tid);
    __syncthreads();
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, std::conditional_t<TA, wmma::col_major, wmma::row_major>> af;
      wmma::load_matrix_sync(af, left + (TA ? k * apitch + wr : wr * apitch + k), apitch);
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, std::conditional_t<TB, wmma::col_major, wmma::row_major>> bf;
        wmma::load_matrix_sync(bf, right + (TB ? (wn + j * 16) * bpitch + k : k * bpitch + wn + j * 16), bpitch);
        wmma::mma_sync(accum[j], af, bf, accum[j]);
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int j = 0; j < 2; ++j)
    wmma::store_matrix_sync(result + wr * BN + wn + j * 16, accum[j], BN, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < BM * BN; i += 128) {
    const int r = row + i / BN, n = column + i % BN;
    if (r < s.rows && n < s.columns) store_result<T>(out, s, r, n, result[i]);
  }
#else
  // A pre-SM80 PTX image may be JIT-compiled on a newer device. Preserve
  // correctness even when that image cannot contain BF16 MMA instructions.
  batch_pointers(s, int(blockIdx.y), a, b, out);
  int row, column;
  tile_origin<32, 64>(s, row, column);
  for (int i = int(threadIdx.x); i < 32 * 64; i += 128) {
    const int r = row + i / 64, n = column + i % 64;
    if (r < s.rows && n < s.columns) {
      float sum = 0;
      for (int k = 0; k < s.shared; ++k)
        sum = fmaf(read(a[TA ? size_t(k) * s.lda + r : size_t(r) * s.lda + k]),
                   read(b[TB ? size_t(n) * s.ldb + k : size_t(k) * s.ldb + n]), sum);
      store_result<T>(out, s, r, n, sum);
    }
  }
#endif
}

// This shared FP16/BF16 path requires SM80 for BF16 WMMA. FP16 could target an
// older MMA generation separately; the common path here deliberately does not.
bool tensor_supported() {
  int device = 0;
  check_cuda_execution(cudaGetDevice(&device), "Native GEMM device");
  static thread_local int cached_device = -1, major = 0;
  if (cached_device != device) {
    check_cuda_execution(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device), "Native GEMM capability");
    cached_device = device;
  }
  return major >= 8;
}

// Local adaptation of [1]'s sliced-K idea: four warps compute the SAME 16x32
// output tile over disjoint K=32 chunks (warp*BK, then strides of 4*BK).
// Smaller output tiles create more blocks for underfilled grids; splitting K
// also gives the four warps independent work. Unlike inter-block split-K, all
// partials stay in shared memory and a fixed pairwise sum needs no atomics,
// second kernel or global workspace. Bias/scaling run once after reduction.
// Warp-private tiles permit __syncwarp inside the uneven-length K loops ([6]);
// the block barrier comes only after every warp has written its partial tile.
template<class T, bool TA, bool TB>
__global__ void tensor_split_gemm(void* out, const T* a, const T* b, Matrix s) {
#if __CUDA_ARCH__ >= 800
  using namespace nvcuda;
  constexpr int BM = 16, BN = 32, BK = 32;
  constexpr int apitch = (TA ? BM : BK) + 8, bpitch = (TB ? BK : BN) + 8;
  __shared__ __align__(32) T left[4][(TA ? BK : BM) * apitch], right[4][(TB ? BN : BK) * bpitch];
  __shared__ __align__(32) float partial[4][BM * BN];
  batch_pointers(s, int(blockIdx.y), a, b, out);
  int row, column;
  tile_origin<BM, BN>(s, row, column);
  const int tid = int(threadIdx.x), warp = tid / 32, lane = tid % 32;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accum[2];
  wmma::fill_fragment(accum[0], 0.f); wmma::fill_fragment(accum[1], 0.f);
  for (int start = warp * BK; start < s.shared; start += 4 * BK) {
    stage_tensor<T, TA ? BK : BM, TA ? BM : BK, apitch, 32>(left[warp], a, s.lda,
        TA ? start : row, TA ? row : start, TA ? s.shared : s.rows, TA ? s.rows : s.shared, lane);
    stage_tensor<T, TB ? BN : BK, TB ? BK : BN, bpitch, 32>(right[warp], b, s.ldb,
        TB ? column : start, TB ? start : column, TB ? s.columns : s.shared, TB ? s.shared : s.columns, lane);
    __syncwarp();
    #pragma unroll
    for (int k = 0; k < BK; k += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, T, std::conditional_t<TA, wmma::col_major, wmma::row_major>> af;
      wmma::load_matrix_sync(af, left[warp] + (TA ? k * apitch : k), apitch);
      #pragma unroll
      for (int j = 0; j < 2; ++j) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, T, std::conditional_t<TB, wmma::col_major, wmma::row_major>> bf;
        wmma::load_matrix_sync(bf, right[warp] + (TB ? j * 16 * bpitch + k : k * bpitch + j * 16), bpitch);
        wmma::mma_sync(accum[j], af, bf, accum[j]);
      }
    }
    __syncwarp();
  }
  #pragma unroll
  for (int j = 0; j < 2; ++j) wmma::store_matrix_sync(partial[warp] + j * 16, accum[j], BN, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < BM * BN; i += 128) {
    const int r = row + i / BN, n = column + i % BN;
    if (r < s.rows && n < s.columns)
      store_result<T>(out, s, r, n, (partial[0][i] + partial[1][i]) + (partial[2][i] + partial[3][i]));
  }
#else
  batch_pointers(s, int(blockIdx.y), a, b, out);
  int row, column;
  tile_origin<16, 32>(s, row, column);
  for (int i = int(threadIdx.x); i < 16 * 32; i += 128) {
    const int r = row + i / 32, n = column + i % 32;
    if (r < s.rows && n < s.columns) {
      float sum = 0;
      for (int k = 0; k < s.shared; ++k)
        sum = fmaf(read(a[TA ? size_t(k) * s.lda + r : size_t(r) * s.lda + k]),
                   read(b[TB ? size_t(n) * s.ldb + k : size_t(k) * s.ldb + n]), sum);
      store_result<T>(out, s, r, n, sum);
    }
  }
#endif
}

// Elman-specific dispatch, guided by [1]'s tile-size/parallelism tradeoff and
// measured by Tests/CudaGemmTest.cu. Decode avoids mostly empty GEMM tiles;
// FP16/BF16 use WMMA or sliced-K; medium/large FP32 uses the pipelined SIMT path.
// The 24/48-block cutoffs and row thresholds were tuned on the 24-SM RTX 4060,
// not computed from device occupancy. They are performance heuristics only;
// every selected kernel masks tails and preserves the same operator semantics.
template<class T, bool TA, bool TB>
void launch(void* out, const void* a, const void* b, Matrix s) {
  if (!s.rows || !s.columns || !s.batches) return;
  const T* left = static_cast<const T*>(a);
  const T* right = static_cast<const T*>(b);
  // Grid Y is limited to 65535. Keep the full batch map when splitting launches.
  for (s.batch_base = 0; s.batch_base < s.batches;) {
    const unsigned batch_count = unsigned(std::min(65535, s.batches - s.batch_base));
    if (!TA && (s.rows <= (std::is_same_v<T, float> ? 16 : 3) || (s.rows <= 64 && s.shared <= 64))) {
      dim3 grid(unsigned(s.rows * ((s.columns + (TB ? 7 : 31)) / (TB ? 8 : 32))), batch_count);
      if constexpr (TB) small_gemm<T, true><<<grid, 256, 0, current_cuda_stream()>>>(out, left, right, s);
      else if (s.rows == 1) small_gemm<T, false><<<grid, 256, 0, current_cuda_stream()>>>(out, left, right, s);
      else small_nn_gemm<T><<<grid, 256, 0, current_cuda_stream()>>>(out, left, right, s);
    } else if constexpr (!std::is_same_v<T, float>) {
      if (s.rows >= 4 && s.columns >= 16 && s.shared >= 16 && tensor_supported()) {
        if (int64_t((s.rows + 31) / 32) * ((s.columns + 63) / 64) * batch_count < 24 && s.shared >= 64) {
          dim3 grid(unsigned((s.rows + 15) / 16 * ((s.columns + 31) / 32)), batch_count);
          tensor_split_gemm<T, TA, TB><<<grid, 128, 0, current_cuda_stream()>>>(out, left, right, s);
        } else {
          dim3 grid(unsigned((s.rows + 31) / 32 * ((s.columns + 63) / 64)), batch_count);
          tensor_gemm<T, TA, TB><<<grid, 128, 0, current_cuda_stream()>>>(out, left, right, s);
        }
      } else {
        dim3 grid(unsigned((s.rows + 15) / 16 * ((s.columns + 63) / 64)), batch_count);
        simt_gemm<T, TA, TB, 16><<<grid, 64, 0, current_cuda_stream()>>>(out, left, right, s);
      }
    } else if (s.rows >= 64 && s.columns >= 64 && s.shared >= 64) {
      const int tiles_m = (s.rows + 127) / 128, tiles_n = (s.columns + 127) / 128;
      // Larger register tiles amortize staging when the grid is wide enough.
      // Useful fraction = M*N / (ceil(M/128)*ceil(N/128)*128*128). Require >=7/8
      // and >=48 blocks; otherwise 64x64 tiles reduce edge waste/grid underfill.
      if (s.rows >= 128 && s.columns >= 128 && int64_t(tiles_m) * tiles_n * batch_count >= 48 &&
          int64_t(s.rows) * s.columns * 8 >= int64_t(tiles_m) * tiles_n * 128 * 128 * 7) {
        dim3 grid(unsigned(tiles_m * tiles_n), batch_count);
        pipelined_fp32_gemm<TA, TB, 128, 128, 8><<<grid, 256, 0, current_cuda_stream()>>>(out, left, right, s);
      } else {
        dim3 grid(unsigned((s.rows + 63) / 64 * ((s.columns + 63) / 64)), batch_count);
        pipelined_fp32_gemm<TA, TB><<<grid, 128, 0, current_cuda_stream()>>>(out, left, right, s);
      }
    } else if (s.rows <= 64) {
      dim3 grid(unsigned((s.rows + 15) / 16 * ((s.columns + 31) / 32)), batch_count);
      simt_gemm<T, TA, TB, 16, 32, 2, 1, 32><<<grid, 256, 0, current_cuda_stream()>>>(out, left, right, s);
    } else {
      dim3 grid(unsigned((s.rows + 63) / 64 * ((s.columns + 63) / 64)), batch_count);
      simt_gemm<T, TA, TB, 64><<<grid, 256, 0, current_cuda_stream()>>>(out, left, right, s);
    }
    check_cuda_execution(cudaGetLastError(), "Native GEMM launch");
    s.batch_base += int(batch_count);
  }
}

template<class T>
void transpose_launch(void* out, const void* a, const void* b, const Matrix& s, bool ta, bool tb) {
  if (ta) {
    if (tb) launch<T, true, true>(out, a, b, s);
    else launch<T, true, false>(out, a, b, s);
  } else {
    if (tb) launch<T, false, true>(out, a, b, s);
    else launch<T, false, false>(out, a, b, s);
  }
}
void dispatch(void* out, const void* a, const void* b, DType dtype, const Matrix& s, bool ta = false, bool tb = false) {
  switch (dtype) {
    case DType::Float32: transpose_launch<float>(out, a, b, s, ta, tb); break;
    case DType::Float16: transpose_launch<__half>(out, a, b, s, ta, tb); break;
    case DType::BFloat16: transpose_launch<__nv_bfloat16>(out, a, b, s, ta, tb); break;
    default: throw std::runtime_error("Unsupported native GEMM dtype");
  }
}
Matrix attention_matrix(const AttentionGemm& shape) {
  Matrix s;
  s.rows = shape.rows; s.columns = shape.columns; s.shared = shape.shared;
  s.lda = s.shared; s.ldb = shape.b_leading ? shape.b_leading : shape.transpose_b ? s.shared : s.columns;
  s.ldc = s.columns; s.batches = shape.batches; s.groups = shape.groups;
  s.stride_a = shape.a_stride; s.stride_b = shape.b_stride; s.stride_out = shape.output_stride;
  return s;
}
float attribute(const KernelParams& p, int index) {
  int bits = p.attribute(index);
  float result;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}
} // namespace

// Full FP32 is also valid when TF32 is permitted. Never narrow FP32 to FP16.
void set_cuda_tf32_math(bool, bool convolution) { convolution_tf32 = convolution; }
bool cuda_conv_tf32_enabled() { return convolution_tf32; }
GemmLibraryVersions gemm_library_versions_cuda() { return {0, 0}; }

bool try_gemm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b,
                   const uint32_t* c, const KernelParams& p, GemmOperation operation) {
  const DType dtype = DType(p.output_dtype());
  if (!is_compute_float(dtype) || p.input_dtype(KernelInput::A) != p.output_dtype() ||
      p.input_dtype(KernelInput::B) != p.output_dtype()) return false;
  if (!p.output_count()) return true;
  const bool linear = operation == GemmOperation::Linear;
  const bool ta = operation == GemmOperation::Matmul && p.attribute(operator_attr::matmul::kTransposeA);
  const bool tb = linear || (operation == GemmOperation::Matmul && p.attribute(operator_attr::matmul::kTransposeB));
  Matrix s;
  s.columns = p.output_shape(p.output_rank() - 1);
  s.rows = linear ? p.output_count() / s.columns : p.output_shape(p.output_rank() - 2);
  s.shared = p.input_shape(KernelInput::A, p.input_rank(KernelInput::A) - (ta ? 2 : 1));
  s.lda = ta ? s.rows : s.shared; s.ldb = tb ? s.shared : s.columns; s.ldc = s.columns;
  s.batches = linear ? 1 : p.output_count() / (s.rows * s.columns);
  s.stride_out = int64_t(s.rows) * s.columns;
  if (operation == GemmOperation::Addmm) {
    s.alpha = attribute(p, operator_attr::addmm::kAlpha);
    s.beta = attribute(p, operator_attr::addmm::kBeta);
  } else if (linear && p.input_rank(KernelInput::C) >= 0) s.beta = 1;
  // Keep the scalar kernels' ignored-NaN and empty-product semantics.
  if (!s.shared || s.alpha == 0) return false;
  if (s.beta != 0) {
    if (p.input_rank(KernelInput::C) < 0 || p.input_dtype(KernelInput::C) != p.output_dtype()) return false;
    s.bias = c;
    const int rank = p.input_rank(KernelInput::C);
    if (rank > 2 || (!linear && s.batches != 1)) return false;
    s.bias_column = rank > 0 && p.input_shape(KernelInput::C, rank - 1) != 1 ? 1 : 0;
    s.bias_row = rank == 2 && p.input_shape(KernelInput::C, 0) != 1 ? p.input_stride(KernelInput::C, 0) : 0;
  }
  if (!linear && s.batches > 1) {
    s.batch_rank = p.output_rank() - 2;
    for (int axis = 0; axis < s.batch_rank; ++axis) {
      s.shape[axis] = p.output_shape(axis);
      const int aa = axis - (p.output_rank() - p.input_rank(KernelInput::A));
      const int ba = axis - (p.output_rank() - p.input_rank(KernelInput::B));
      if (aa >= 0 && p.input_shape(KernelInput::A, aa) != 1) s.a_step[axis] = p.input_stride(KernelInput::A, aa);
      if (ba >= 0 && p.input_shape(KernelInput::B, ba) != 1) s.b_step[axis] = p.input_stride(KernelInput::B, ba);
    }
  }
  dispatch(out, a, b, dtype, s, ta, tb);
  return true;
}

void attention_gemm_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, void*, const AttentionGemm& s) {
  dispatch(out, a, b, s.dtype, attention_matrix(s), false, s.transpose_b);
}
void attention_gemm_cuda_pointers(void* pointers, const AttentionGemm& s) {
  Matrix matrix = attention_matrix(s);
  matrix.pointers = static_cast<void* const*>(pointers);
  dispatch(nullptr, nullptr, nullptr, s.dtype, matrix, false, s.transpose_b);
}
void launch_low_attention_matrix_cuda(uint32_t* out, const uint32_t* a, const uint32_t* b, const KernelParams& p) {
  Matrix s;
  const bool ta = p.attribute(0), tb = p.attribute(1);
  s.rows = p.output_shape(2); s.columns = p.output_shape(3);
  s.shared = p.input_shape(KernelInput::A, ta ? 2 : 3);
  s.lda = ta ? s.rows : s.shared; s.ldb = tb ? s.shared : s.columns; s.ldc = s.columns;
  s.batches = p.output_shape(0) * p.output_shape(1);
  s.stride_a = int64_t(s.rows) * s.shared; s.stride_b = int64_t(s.columns) * s.shared;
  s.stride_out = int64_t(s.rows) * s.columns;
  dispatch(out, a, b, DType::Float32, s, ta, tb);
}
bool try_biased_float64_gemm_cuda(double*, const double*, const double*, const double*, int, int, int, bool, double) {
  return false;
}
bool try_conv2d_gemm_cuda(void* output, const void* columns, const void* weights, DType dtype,
                         int positions, int channels, int shared, int input_pitch, int output_pitch,
                         bool output_float32, int groups) {
  Matrix s;
  s.rows = channels; s.columns = positions; s.shared = shared;
  s.lda = shared; s.ldb = input_pitch; s.ldc = output_pitch; s.batches = groups;
  s.stride_a = int64_t(channels) * shared; s.stride_b = int64_t(shared) * input_pitch;
  s.stride_out = int64_t(channels) * output_pitch; s.output_float = output_float32;
  dispatch(output, weights, columns, dtype, s);
  return true;
}
void convolution_filter_partials_cuda(float* partials, const float* packed, int patches, int outputs, int tile, int tiles) {
  Matrix s;
  s.rows = outputs; s.columns = patches; s.shared = tile;
  s.lda = tile; s.ldb = tile; s.ldc = patches; s.batches = tiles;
  s.stride_a = s.stride_b = int64_t(patches + outputs) * tile;
  s.stride_out = int64_t(patches) * outputs;
  dispatch(partials, packed + size_t(patches) * tile, packed, DType::Float32, s, false, true);
}
bool try_pointwise_conv_gradient_cuda(CudaConvolution, uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&) {
  return false;
}
} // namespace elman
