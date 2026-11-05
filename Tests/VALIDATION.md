# Local validation record

Date: 2026-11-03 (Asia/Shanghai).

Environment: Windows x64, Visual Studio 2022 / MSVC 19.44, CUDA Toolkit 13.3,
NVIDIA GeForce RTX 4060, CMake 3.24.

The CPU and CUDA dispatch lists each contain 86 operators.
All 172 native operator files use named `KernelParams` accessors, named
operator attributes, and readable input/output operations. The shared
operator helpers and axis constants are also named. No operator reads a raw
packed ABI slot. `clang-format` 19.1.5 accepted all 179 C++ and CUDA files
under `Src/Operators` against the repository's Google-based style
configuration. `git diff --check HEAD` passed.

Local CI ran after each of the three refactoring milestones: named tensor
metadata, named operator attributes, and readable input/output operations.
At each milestone, the CPU-only Visual Studio Release build and all three
CTest cases passed, and the CUDA-enabled Visual Studio Release build and all
five CTest cases passed. The tests cover native kernel reference values and
CPU/CUDA parity, public tensor/autograd behavior, and model/JSON/checkpoint
behavior. No changes were pushed to a remote.

The inference module extraction ran in two additional milestones. First,
configuration, CSV dataset loading, and classification loss moved to the shared
runtime layer. Second, the inference entry point, checkpoint loading, and
prediction report moved to `Src/Inference`. After each milestone, the CPU
Release build and all three CTest cases passed; the CUDA Release build and all
five CTest cases passed. The model/IO test now exercises the inference module
directly and checks its predictions against the model's forward output.

The autograd extraction ran in three further milestones: gradient accumulation
and backward execution moved to `Src/Core/Autograd.cpp`; operator derivative
rules moved to `Src/Core/AutogradOps.cpp`; and legacy wrapper macros were
replaced with explicit functions. `Src/Core/Tensor.cpp` implements tensor
storage and backend dispatch. After each milestone, the CPU Release build and
all three CTest cases passed, and the CUDA Release build and all five CTest
cases passed. The tensor tests cover gradient accumulation, graph export,
dropout gradients, and mutation guards. No changes were pushed to a remote.

The latest three milestones separated graph recording, topological traversal,
and graph export into `Src/Core/Graph.cpp`; renamed non-operator source, test,
model, and configuration files to PascalCase; and shortened the root README.
After each milestone, the CPU Release build and all three CTest cases passed,
and the CUDA Release build and all five CTest cases passed. Git reported 25
case-only tracked-file renames. All 233 quoted source includes matched the
actual case of repository paths, and both run configurations resolved their
renamed model files. The `Src/Operators` filenames were left unchanged. No
changes were pushed to a remote.

The optimizer extraction ran in two milestones. First, shared gradient
validation, clipping, and state handling moved to `Src/Optimizer/Optimizer.cpp`,
with separate `Sgd.cpp`, `Adam.cpp`, and `AdamW.cpp` update implementations.
The CPU build and three tests passed; the CUDA build and five tests passed.
Second, direct optimizer tests were added for CPU and CUDA. They check the
three algorithms' parameter and moment updates, SGD momentum across steps,
and gradient clipping. The CPU build and four tests passed; the CUDA build
and seven tests passed. No changes were pushed to a remote.

Full-size AlexNet and GPT2 smoke runs require the sample CSV files under
`Data/`. Those files are absent from this checkout, so the full-model cases were
not registered or run. CMake registers them when the corresponding files exist.
Linux and macOS builds still require validation on their native hosts.
