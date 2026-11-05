# Elman

Elman is a compact deep learning framework with CPU and CUDA execution. It
provides tensor operations, a dynamic computation graph, reverse mode automatic
differentiation, model training, checkpoint loading, and inference.

## Capabilities

- Elementwise, reduction, indexing, matrix, normalization, loss, convolution,
  and pooling operations.
- Automatic gradients with graph mutation checks.
- AlexNet and GPT2 model definitions, SGD and Adam family optimizers, and
  checkpoint based inference.
- A backend interface that keeps tensor and model code independent of the
  native CPU and CUDA implementations.

## Code layout

| Path | Purpose |
| --- | --- |
| `Src/Core` | Tensor storage, backend interface, computation graph, and autograd. |
| `Src/Operators` | Native CPU and CUDA operator implementations and ABI definitions. |
| `Src/Backends` | Device buffers, transfers, and operator dispatch. |
| `Src/Models` | AlexNet and GPT2 model construction. |
| `Src/Training` | Training loop and learning rate schedule. |
| `Src/Optimizer` | SGD, Adam, AdamW, and shared gradient handling. |
| `Src/Inference` | Checkpoint based prediction. |
| `Src/Runtime` | Configuration and dataset handling shared by training and inference. |
| `Src/IO` | JSON and checkpoint serialization. |
| `Models`, `Configs` | Model definitions and run configurations. |
| `Tests` | Numerical, model, and integration checks. |

See the [model format](Models/Format.md), [training configuration](Configs/README.md),
and [operator ABI](Src/Operators/ABI.md) for details.
