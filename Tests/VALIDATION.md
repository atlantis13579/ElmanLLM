# Full-model validation record

Date: 2026-10-30 (Asia/Shanghai).

Environment: Windows x64, Visual Studio 2022 / MSVC 19.44, CUDA Toolkit 13.3,
NVIDIA GeForce RTX 4060 with 8 GB memory, CMake 3.24.

## Full-size CUDA models

The configurations use `Models/alexnet.json` and `Models/gpt2.json` directly.
Architecture widths, layer counts, output classes, vocabulary and position
embedding capacity are unchanged. Every parameter participates in training;
there are no frozen backbones, adapters or reduced model variants.

Each model trains on one synthetic sample, with batch size one, one optimizer
step, Float32 arithmetic and its configured dropout (AlexNet 0.5, GPT2 0.1).
AlexNet receives a full 3 x 224 x 224 image. GPT2 receives an eight-token data
sequence while retaining all 1,024 learned positions and the complete vocabulary.
Weights are randomly initialized; this is process validation, not pretraining.

| Model | Parameters | Initial evaluation loss | Final evaluation loss | Updated tensors | Checkpoint output error |
|---|---:|---:|---:|---:|---:|
| AlexNet | 61,100,840 | 8.48581123 | 7.72357035 | 16 / 16 | 0 |
| GPT2 | 124,439,808 | 10.83514786 | 8.72915268 | 148 / 148 | 0 |

All parameter gradients were present and finite. Training saved a versioned
checkpoint with optimizer state, reloaded it and reproduced inference output
exactly. A separate CLI inference process then loaded each checkpoint and
produced finite predictions/loss. Optimizer updates remain host-based; tensor
forward and backward calculations execute native CUDA kernels.

The AlexNet train/save/reload/infer integration test passed in 14.56 seconds;
GPT2 passed in 29.55 seconds on this machine. These timings include initialization
and file I/O and are not throughput benchmarks.

## Automated checks

`ctest --preset windows-cuda` passed 9/9 tests in 50.27 seconds:

- All 86 native entry points on CPU, CUDA and Compute, with independently
  specified values, backend parity and convolution finite differences.
- Public tensor/autograd tests: 204 CPU checks, 212 CUDA checks and 208 Compute
  checks. CUDA includes a complete 50,257-class loss/gradient comparison against
  an independent host reference.
- Model/IO tests: 850 checks for JSON, causal masking, model topology,
  checkpoint restore and corruption rejection.
- Full AlexNet and GPT2 CUDA train/checkpoint/inference tests, serialized to
  bound GPU memory, with exact expected parameter-count assertions.

The Visual Studio 2022 CUDA build was also rebuilt and passed all seven
operator/model/IO regression cases. Full-model execution results above are from
the Ninja CUDA executable.

Full-model CUDA tests have the `full-model` label and a 1,200-second timeout.
They require finite losses/gradients and parameter updates; a single stochastic
update is not required to decrease evaluation loss. Both recorded runs did.
CPU/Compute operator tests remain enabled; full-model training on those
reference backends is not part of this validation.

## Changes needed for full scale

CUDA cross entropy now computes each vocabulary row with a cooperative block
reduction. Its derivative has linear work in vocabulary size instead of
recomputing a full row for every output element. Original per-operator file
names and the scalar reference implementations for other backends are retained.
The wrapper generator preserves these hand-written CUDA implementations.
Checkpoint weight and optimizer arrays use buffered reads/writes while keeping
the existing little-endian format and integrity checksum unchanged.

## Artifacts and platform scope

Raw logs, full checkpoints, metrics, graph exports and independent inference
results are under `Intermediate/CTest/alexnet-full-cuda` and
`Intermediate/CTest/gpt2-full-cuda`. Checkpoints/results are also copied to the
configured `Intermediate/Runs/alexnet_full` and `Intermediate/Runs/gpt2_full`
destinations so the normal inference commands can immediately use them.
Generated data artifacts and binaries are ignored by Git; only the small source
CSV fixtures are versioned.

Metal is implemented and source-reviewed but remains untested on an Apple host.
Linux/Make and macOS/Xcode presets likewise require native host validation.
No model-quality or full-scale throughput claims are made.
