# Elman

Elman is a self-contained C++17 deep-learning training and inference framework.
It includes tensor operations, reverse-mode automatic differentiation, an eager
execution graph, JSON model definitions, optimizers and a command-line runner.
No third-party framework, BLAS library, JSON library or downloaded dataset is
required. Platform SDKs provide CUDA, Direct3D Compute Shader and Metal access.

## Operators and automatic differentiation

All requested operator families are present: elementwise arithmetic, comparisons,
logical operations, reductions, layout/indexing, random generation, casts,
broadcast matrix multiplication, activation, normalization, losses, convolution
and pooling. There are 86 native entry points per backend, including backward
helpers. First-party scalar algorithms are shared through `Src/Kernels/*.inc`.
The hand-written CUDA cross-entropy files use row reductions for the complete
GPT2 vocabulary.

Tensor arithmetic and backward kernels run on the selected backend. Unavailable
backends and mixed-backend operations raise errors; no CPU fallback is hidden.
CUDA/Compute buffers hold device data, and Metal uses native shared buffers.
Some compound public operations, including normalization, are assembled from
primitive device operations so their full gradients are recorded.

The eager graph records parent edges, topological backward callbacks and storage
versions. Gradients accumulate in leaves until `zero_grad()`. Mutation of a graph
input or saved result, including through a reshape/detach alias, is rejected
before backward. Graph export is diagnostic JSON; it is separate from the
versioned model definition and binary checkpoint.

## Models and training

`Models/alexnet.json` and `Models/gpt2.json` are the actual models used by the
training/inference checks: 61,100,840 and 124,439,808 parameters respectively.
AlexNet takes a full 224 x 224 RGB image; GPT2 retains 12 layers, width 768,
12 heads, its 50,257-token vocabulary and all 1,024 position embeddings.
There are no reduced architecture variants. The models start with random weights.

Training supports SGD with momentum, Adam, AdamW, gradient clipping, weight decay,
warmup and cosine schedules. Checkpoints preserve named parameters and optimizer
state with version/shape validation, a checksum and replacement through a
temporary file. Inference uses loaded weights without dropout or gradient
recording.

See [model format](Models/FORMAT.md), [training settings](Configs/README.md),
[data format](Data/README.md), [kernel ABI](Src/Kernels/ABI.md) and
[backend details](Src/Backends/README.md).

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
