# CUDA GEMM without cuBLAS

`ELMAN_ENABLE_CUDA=ON` enables CUDA. `ELMAN_ENABLE_CUBLAS` defaults to `ON`;
setting it to `OFF` selects `CudaGemm.cu` instead of `CudaGemm_cuBLAS.cu` and
removes both cuBLAS and cuBLASLt from the link. No BLAS headers, runtime loading,
or external GEMM package are used by the native implementation. cuDNN is also
disabled in this configuration because it can load cuBLAS transitively.
CUDA runtime, driver and the existing NVRTC fusion dependencies remain.

The native backend supports FP32, FP16 and BF16 matmul, batched/broadcast matmul,
linear and addmm. Attention (including grouped and device-pointer batches) and
convolution's GEMM operations use the same backend. Existing scalar kernels
retain other types and exceptional/empty-product behavior. FP64 retains its
existing CUDA reference implementation. FP32 uses full precision even if TF32
is permitted; enabling TF32 does not narrow operands or change results.

## Build and run

```powershell
cmake --preset windows-vs2022-cuda-lite
cmake --build --preset windows-vs2022-cuda-lite
ctest --preset windows-vs2022-cuda-lite -R "cuda_gemm|_cuda$|cuda_math_modes"
Binaries/Windows/vs2022-cuda-lite/Release/elman_cuda_gemm_tests.exe --benchmark
Binaries/Windows/vs2022-cuda-lite/Release/elman_cuda_gemm_tests.exe --benchmark-fp32
```

`windows-cuda-lite` provides the equivalent Ninja preset in a developer command
prompt; `linux-cuda-lite` provides the Linux preset. For a custom build:

```sh
cmake -S . -B Build/cuda-lite -DELMAN_ENABLE_CUDA=ON -DELMAN_ENABLE_CUBLAS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build Build/cuda-lite --parallel 6
```

Select architectures appropriate for the deployment GPU. Native FP16/BF16 MMA
requires an SM80-or-newer image and device; older targets use CUDA arithmetic.
Use a separate build/output directory for each backend when comparing sizes or
performance. The ordinary CUDA presets remain cuBLAS-enabled by default.

## Kernel design

- FP32: medium and large products use 64x64 or 128x128 output tiles, K=8 double
  buffered shared-memory stages, and 8x4 or 8x8 per-thread register accumulators.
  Warp tiles reuse operands across registers; global loads prefetch the next
  stage while the current stage computes, and two register fragments overlap
  shared loads with FP32 FMA. Aligned float4 shared reads use 32-column strips
  to avoid the bank conflicts of contiguous 8-column thread tiles. The larger
  tile requires at least 48 blocks and at least seven-eighths useful work after
  padding; smaller grids and irregular edges use 64x64 tiles. Small products
  retain the original K=32 SIMT tiles or vector reductions. All four transpose
  specializations load the stored orientation without materializing a transpose.
- FP16/BF16: 32x64 output tiles, K=32 shared staging and warp-level 16x16x16 MMA.
  Staging copies eight elements per 128-bit transaction in the stored orientation.
  Underfilled grids use 16x32 output tiles with four warps splitting K and a
  deterministic shared-memory reduction. Accumulation and the fused
  bias/alpha/beta epilogue use FP32.
- Decode: small row counts use vectorized warp reductions for transposed weights
  or a K reduction across warps for ordinary row-major weights. FP16/BF16 can
  switch to Tensor Cores starting at four rows.
- Tiles are grouped along M to improve B reuse. Bounds masks handle arbitrary
  tails and staging removes global-pointer alignment requirements. Broadcast
  offsets and attention pointer arrays are resolved on the GPU. Launches use the
  active execution stream, require no workspace and support graph capture.

Design references: [CUTLASS efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html),
[CUTLASS SIMT pipelining](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/gemm/threadblock/mma_pipelined.h)
and [Triton matrix multiplication](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html).
The kernels are written for Elman's ABI; neither project is a dependency.

## Validation and measurement

`cuda_gemm` checks all four transpose combinations, odd dimensions, tail tiles,
unaligned pointers, output guards, zero dimensions, alpha/beta ignored-NaN
semantics, scalar/vector/matrix addends, broadcast forward/backward, grouped
attention with padded leading dimensions, non-monotonic device pointer batches,
and batch counts crossing the CUDA grid-Y limit. Both pipelined tile sizes are
checked with full-output references, epilogues, padded attention strides and
device pointer batches. Cancellation cases retain FP32 mantissa bits below
TF32 precision; large/small exponent products catch accidental FP16 narrowing.
References use CPU double
accumulation over the values actually representable in each input dtype.
Existing CUDA operator, attention, graph, dtype and math-policy suites check
integration with the rest of the framework.
`cuda_no_blas_dependencies` inspects the built library's imports when `dumpbin`
or `readelf` is available. It rejects cuBLAS, cuBLASLt and cuDNN dependencies.

`--benchmark` first runs correctness checks, then prints CSV latency and GFLOP/s
for decode, small-model projections, larger GEMM and irregular sizes. Timed
inputs are aligned; a CPU reference checks sampled outputs of every timed shape.
Both backends use TF32 off, identical inputs, fused linear bias, ten warmups,
32 operations per CUDA Graph, at least 150 ms of graph warmup per shape and seven samples of sixteen
replays. The reported latency is the median CUDA-event time per operation.
Transfers, allocation, graph construction and CPU reference evaluation are not
timed. This suite reuses cache-resident inputs: these are steady-state GPU kernel
measurements, not end-to-end model latency or cold-weight bandwidth.

`--benchmark-fp32` measures 48 FP32 cases: square products through 2048 cubed,
wide/tall products, irregular tails, and all NN/NT/TN/TT layouts. For products
above two billion FLOPs it uses four operations per graph and two replays per
sample; all other settings are identical. The larger working sets can exceed
L2, although allocations are reused. Correctness uploads are explicitly ordered
on the test's nonblocking execution stream; the CUDA runtime's
[pageable-copy synchronization rules](https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html)
do not guarantee completion before kernels on a different stream.

Run the same executable from a cuBLAS-enabled build for comparison. Performance
depends on shape, dtype, GPU and clock state; larger GEMMs can favor cuBLAS.

## Measured results

Measured on 2026-11-26 with an RTX 4060 (SM89, WDDM), driver 621.01, CUDA 13.3.73,
cuBLAS/cuBLASLt 13.6.0 and MSVC 19.44 Release. The complete 48-case comparison is
in [CudaGemmBenchmark.csv](CudaGemmBenchmark.csv); `native_over_cublas` below 1
means the native backend is faster. These are GPU timings under the protocol
above, not a claim of universal speedup.

| Type | Operation | M x N x K | cuBLAS (us) | Native (us) |
| --- | --- | --- | ---: | ---: |
| FP32 | Linear + bias | 1 x 512 x 512 | 7.152 | 2.272 |
| FP32 | Matmul | 37 x 71 x 59 | 4.713 | 2.062 |
| FP32 | Matmul | 64 x 256 x 256 | 4.856 | 10.661 |
| FP32 | Matmul | 512 x 512 x 512 | 40.742 | 47.580 |
| FP16 | Linear + bias | 1 x 512 x 512 | 4.428 | 2.275 |
| FP16 | Matmul | 4 x 1024 x 1024 | 4.842 | 10.190 |
| FP16 | Matmul | 64 x 256 x 256 | 2.641 | 3.626 |
| FP16 | Matmul | 512 x 512 x 512 | 15.174 | 17.170 |

### Large FP32 optimization

[CudaGemmFP32Benchmark.csv](CudaGemmFP32Benchmark.csv) compares the native kernel
at commit `27b91d6` with the pipelined kernel and cuBLAS, with TF32 disabled in
every build. Each cell below is the median of three independent executions of
the seven-sample protocol. The CSV includes all three run medians for each
backend so desktop-GPU variation remains visible. All 48 shape/layout cases
improved over the original native kernel: median speedup 1.98x, range 1.51x to
2.34x. These measurements cover one RTX 4060, not other GPU architectures.

| M x N x K | Layout | Before (us) | After (us) | cuBLAS (us) | Speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| 512 x 512 x 512 | NN | 88.460 | 43.902 | 42.645 | 2.02x |
| 1024 x 1024 x 1024 | NN | 596.860 | 273.664 | 283.120 | 2.18x |
| 1536 x 1536 x 1536 | NN | 1940.992 | 829.516 | 901.212 | 2.34x |
| 2048 x 2048 x 2048 | NN | 4625.792 | 2052.480 | 2295.384 | 2.25x |
| 2048 x 2048 x 2048 | NT | 4791.296 | 2158.440 | 2299.008 | 2.22x |
| 2048 x 2048 x 2048 | TN | 4615.040 | 2089.720 | 1836.416 | 2.21x |
| 2048 x 2048 x 2048 | TT | 4626.920 | 2278.500 | 2628.096 | 2.03x |
| 256 x 4096 x 512 | NN | 307.526 | 145.048 | 169.214 | 2.12x |
| 1025 x 769 x 257 | NN | 163.760 | 81.926 | 64.846 | 2.00x |

The native path is useful when removing the BLAS deployment dependency matters.
Large square FP32 performance is now much closer to cuBLAS on this GPU; some
transpose layouts, irregular shapes and small-batch projections remain slower.
FP16/BF16 benefit from Tensor Cores without sacrificing FP32 accumulation.

The local cuBLAS and cuBLASLt DLLs total 492.44 MiB. The native core is 21.48 MiB
versus 20.20 MiB for the cuBLAS core. The default build resolves cuBLAS from the
Toolkit rather than copying it beside the executable, so the reduction is in
required deployment dependencies, not an existing bundled DLL directory.
The approximately 103.07 MiB of NVRTC/builtins remains in both configurations.

Validation of this build:

- GEMM correctness, dependency imports, math policy and workspace-free memory
  accounting pass. The cuBLAS build also passes the GEMM, math-policy and memory
  checks.
- Compute Sanitizer memcheck: 0 errors; racecheck: 0 errors and 0 warnings.
- SM89 resource inspection shows no local-memory or stack allocation in any
  pipelined specialization. The 64x64 kernels use 79-89 registers and 8 KiB of
  shared memory; the 128x128 kernels use 126-133 registers and 16 KiB.
- The GEMM executable runs with CUDA Toolkit entries removed from `PATH`.
- Native CUDA integration: 33/33 tests pass, including fusion, graph training,
  SDPA, paged/tiled attention, tensor/kernel gradients, convolution and dtype
  suites. The cuBLAS build also passes all 32 CUDA integration tests. FP8/FP4
  attention passes its 268 checks with both GEMM backends; native attention
  also passes Compute Sanitizer memcheck with zero errors.
- Fusion treats tagged FP32 attention intermediates as dedicated operations,
  rather than ordinary scaled/masked softmax, and releases a drained region's
  stream so its BLAS workspace can be destroyed. Regression tests compare
  FP8/FP4 forward/backward results with fusion enabled/disabled, both with and
  without runtime verification; stream lifetime checks cover synchronous,
  asynchronous and exception-fallback execution.
