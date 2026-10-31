# Elman

Elman is a self-contained C++17 deep-learning training and inference framework.
It includes tensor operations, reverse-mode automatic differentiation, an eager
execution graph, JSON model definitions, optimizers and a command-line runner.
No third-party framework, BLAS library, JSON library or downloaded dataset is
required. Platform SDKs provide CUDA, Direct3D Compute Shader and Metal access.

## Repository layout

- `Src`: runtime, operators, automatic differentiation, model and training code.
- `Models`: versioned JSON model definitions.
- `Configs`: JSON training and inference settings.
- `Data`: small deterministic smoke-test fixtures.
- `Tests`: numerical correctness and integration checks.
- `Build`: generated CMake projects (ignored).
- `Intermediate`: objects, logs and checkpoints (ignored).
- `Binaries`: final executable and library outputs (ignored).

Visual Studio objects, compiler databases and tracking files use the configured
`Intermediate/<preset>` path. Ninja/Make target build directories also live there;
their top-level project files remain in `Build`. CMake's small compiler-detection
diagnostics may remain in `Build/CMakeFiles`. Xcode uses `Intermediate/Xcode`.

## Build and run on Windows

Requirements: CMake 3.24+, Visual Studio 2022 with C++ tools and a Windows SDK.
CUDA additionally requires the NVIDIA CUDA Toolkit and a compatible driver.
The project only uses the CUDA runtime, without cuBLAS, cuDNN or other math
libraries. CUDA presets target compute capability 8.9 for the local RTX 4060;
override `-DCMAKE_CUDA_ARCHITECTURES=...` for a different GPU.

From a Visual Studio x64 developer command prompt at the repository root:

```text
cmake --preset windows-cuda
cmake --build --preset windows-cuda
ctest --preset windows-cuda
Binaries\Windows\windows-cuda\elman.exe devices
Binaries\Windows\windows-cuda\elman.exe train Configs\alexnet_smoke.json
Binaries\Windows\windows-cuda\elman.exe train Configs\gpt2_smoke.json
Binaries\Windows\windows-cuda\elman.exe infer Configs\gpt2_smoke.json
```

For a Visual Studio solution, configure `windows-vs2022` for CPU and Compute,
or `windows-vs2022-cuda` for all three Windows backends. The CUDA solution
requires the toolkit's Visual Studio integration. Open the generated solution
under `Build/vs2022` or `Build/vs2022-cuda`, or use the matching build preset.
Visual Studio Release executables are under `Binaries/Windows/vs2022/Release`
or `Binaries/Windows/vs2022-cuda/Release`. Each preset has its own output directory
to prevent CPU and CUDA builds from overwriting one another.

```text
cmake --preset windows-vs2022
cmake --build --preset windows-vs2022
ctest --preset windows-vs2022
```

## Linux and macOS

Linux requires a C++17 compiler and Make; use `linux-cuda` to enable CUDA.
macOS requires Xcode and its command-line tools. These platform presets are
provided but were not executed on the Windows development machine.

```text
cmake --preset linux
cmake --build --preset linux
ctest --preset linux
./Binaries/Linux/linux/elman train Configs/gpt2_smoke.json --backend cpu

cmake --preset macos-xcode
cmake --build --preset macos-xcode
ctest --preset macos-xcode
./Binaries/Darwin/xcode/Release/elman train Configs/gpt2_smoke.json --backend metal
```

## Operators and automatic differentiation

All requested operator families are present: elementwise arithmetic, comparisons,
logical operations, reductions, layout/indexing, random generation, casts,
broadcast matrix multiplication, activation, normalization, losses, convolution
and pooling. There are 86 native entry points per backend, including backward
helpers. Each has its own `operator_cpu.cpp`, `operator_cuda.cu`,
`operator_compute.hlsl` and `operator_metal.metal` file under `Src/Operators`.
First-party scalar algorithms are shared through `Src/Kernels/*.inc`; the
checked-in wrappers need no Python during builds. `Src/Kernels/generate.py`
regenerates the wrappers and dispatch registries when developing operators.
The hand-written CUDA cross-entropy files use row reductions for the complete
GPT2 vocabulary and are preserved by the generator.

Tensor arithmetic and backward kernels run on the selected backend. Unavailable
backends and mixed-backend operations raise errors; no CPU fallback is hidden.
CUDA/Compute buffers hold device data, and Metal uses native shared buffers.
Some compound public operations, including normalization, are assembled from
primitive device operations so their full gradients are recorded.

```cpp
#include "Core/tensor.h"
using namespace elman;
Tensor x = Tensor::from_data({2}, {2, 3}, Backend::CUDA, true);
Tensor loss = sum_all(mul(x, x));
loss.backward();
auto gradient = x.grad(); // {4, 6}
auto graph = loss.graph_json();
```

The eager graph records parent edges, topological backward callbacks and storage
versions. Gradients accumulate in leaves until `zero_grad()`. Mutation of a graph
input or saved result, including through a reshape/detach alias, is rejected
before backward. Graph export is diagnostic JSON; it is separate from the
versioned model definition and binary checkpoint.

## Models, training and validation

`Models/alexnet.json` and `Models/gpt2.json` are the actual models used by the
training/inference checks: 61,100,840 and 124,439,808 parameters respectively.
AlexNet takes a full 224 x 224 RGB image; GPT2 retains 12 layers, width 768,
12 heads, its 50,257-token vocabulary and all 1,024 position embeddings.
There are no reduced architecture variants. Each smoke configuration uses one
sample, batch size one and one optimizer update with the model's normal dropout.
GPT2's data sequence is eight tokens long without changing its model definition
or parameter count. The models start with random weights; these runs validate
the complete process and do not produce useful pretrained models.

See [model format](Models/FORMAT.md), [training settings](Configs/README.md),
[data format](Data/README.md), [kernel ABI](Src/Kernels/ABI.md) and
[backend details](Src/Backends/README.md). Executed checks and results are recorded
in [validation](Tests/VALIDATION.md).

Training supports SGD with momentum, Adam, AdamW, gradient clipping, weight decay,
warmup and cosine schedules. The console reports loss, gradient norm, learning
rate and elapsed time. Checkpoints preserve named parameters and optimizer state
with version/shape validation, a checksum and replacement through a temporary
file. Inference uses loaded weights without dropout or gradient recording.

CTest covers all 86 native entry points, CPU/backend parity, analytical versus
finite-difference gradients, grouped convolution gradients, causal GPT2 attention,
JSON rejection and checkpoint corruption. CUDA integration cases train both
full-size models, assert the exact parameter counts, finite gradients, parameter
updates and checkpoint equivalence, then run inference in a separate process.
Each run has its own directory under `Intermediate/CTest`. Full-model tests are
serialized to bound GPU memory usage. One stochastic update is not required to
lower evaluation loss; loss values are still recorded.

## First-version scope

- Numerical training uses Float32. Cast/storage types are Float32, Int32 and
  Bool32; arithmetic is float-based, and float host-transfer APIs cannot represent
  every Int32 value exactly. Integer-preserving layout copies retain raw words.
- Tensors are contiguous, have up to eight positive dimensions and use 32-bit
  indexing. Reshape shares storage; permute and other layout operations copy.
  Empty tensors and negative slice steps are not supported.
- Reduction `dim=-1` means the last axis; `sum_all` and `mean_all` reduce all
  elements. Extrema return values and split gradients across tied extrema.
- Layer/RMS normalization currently normalizes the last dimension. Batch
  normalization uses current-batch statistics; persistent running statistics
  are not part of this API. Group normalization supports channel groups.
- Reference kernels prioritize correctness over throughput. Optimizer updates
  and index validation currently transfer data through the host. There is no
  mixed precision, distributed training, kernel fusion or vendor BLAS.
- JSON model version 1 configures the AlexNet and GPT2 architecture families.
  C++ callers can compose other models using the tensor API. GPT2 inference
  reports teacher-forced token predictions; tokenizer and text generation are
  outside this first version.
- Resume restores parameters, moments and the global step. Dataset shuffle
  position and learning-rate schedule phase restart for the new run. Checkpoints
  include optimizer moments, while exact shuffled/scheduled continuation is not
  part of this first version.
- Metal is implemented and source-reviewed, but needs compilation and execution
  on macOS. Linux and Xcode builds likewise require validation on their hosts.

## Local milestones

The repository is committed locally in milestones covering architecture,
four-backend operators/autograd, models/CLI, and integration verification.

All milestone commits are local. Generated binaries, build trees and training
artifacts are excluded from version control. Metal and Xcode require a macOS
host; source availability is reported separately from executed validation.
