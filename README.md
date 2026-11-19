# Elman

Elman is a compact deep learning framework with CPU, CUDA, and Metal execution. It
provides tensor operations, a dynamic computation graph, reverse mode automatic
differentiation, model training, checkpoint loading, and inference.

## Capabilities

- Elementwise, reduction, indexing, matrix, normalization, loss, convolution,
  and pooling operations.
- Automatic gradients with graph mutation checks.
- Linear/logistic regression, AlexNet and GPT2 model definitions, SGD and Adam family optimizers, and
  checkpoint based inference.
- Native Qwen2 inference with official DeepSeek-R1-Distill-Qwen-1.5B weights,
  grouped-query attention, rotary positions, KV caching, greedy generation,
  UTF-8 streaming and stop strings.
- Affine INT4 weights, sampled generation, paged KV storage, chunked prefill,
  continuous batching and prefix reuse.
- A built-in HTTP service with a browser workspace for streaming chat and image
  generation with Stable Diffusion or FLUX.2 Klein.
- Native DeepSeek latent attention and sparse experts, layer placement across
  CUDA devices, single-card partition simulation, and complete Qwen decode CUDA
  Graphs.
- Independently trainable byte BPE tokenization and an official Hugging Face
  tokenizer/chat-template import path for DeepSeek.
- A backend interface that keeps tensor and model code independent of the
  native implementations.

## Command line

Run these commands from the repository root with `elman` on your `PATH`.

```sh
elman --help
elman devices
```

### Serving

Set `model.path` in each serving configuration to the local checkpoint
directory. Relative paths resolve from the JSON file's directory. The
examples select CUDA by default. Set `model.backend` to an available backend
reported by `elman devices`.

```sh
# DeepSeek chat at http://127.0.0.1:8080
elman --server-config Configs/Serving/DeepSeekServer.json

# Stable Diffusion at http://127.0.0.1:8081
elman --server-config Configs/Serving/StableDiffusionServer.json

# FLUX.2 Klein at http://127.0.0.1:8082
elman --server-config Configs/Serving/Flux2Server.json
```

Run each model in a separate terminal. The FLUX.2 configuration uses the
Apache 2.0 checkpoint. Download its weights and supporting files with the
Hugging Face CLI before starting the server. [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

```sh
hf download black-forest-labs/FLUX.2-klein-4B --revision e7b7dc27f91deacad38e78976d1f2b499d76a294 --local-dir Models/FLUX.2-klein-4B --include "LICENSE.md" --include "flux-2-klein-4b.safetensors" --include "text_encoder/*" --include "tokenizer/*" --include "vae/*"
```

The FLUX.2 web workspace defaults to 512 × 512 and four denoising steps.
The `POST /sdapi/v1/txt2img` endpoint also accepts a prompt and optional
width, height, steps, and seed. The image dimensions must be divisible by 16.

### Training and evaluation

Set `data.path` in the training configuration to the dataset CSV before running
these commands. `infer` loads the checkpoint written by `train`.

```sh
elman train Configs/Training/LinearRegression.json
elman infer Configs/Training/LinearRegression.json
```

## Code layout

| Path | Purpose |
| --- | --- |
| `Src/Core` | Tensor storage, backend interface, computation graph, and autograd. |
| `Src/Operators` | Native CPU, CUDA, and Metal operator implementations and ABI definitions. |
| `Src/Backends` | Device buffers, transfers, and operator dispatch. |
| `Src/Models` | Trainable models and Qwen2/DeepSeek inference decoders. |
| `Src/Training` | Training loop and learning rate schedule. |
| `Src/Optimizer` | SGD, Adam, AdamW, and shared gradient handling. |
| `Src/Inference` | Generation, caches, sampling, memory admission and execution placement. |
| `Src/Serving` | Model request scheduling. |
| `Apps` | Cross-platform executable and application services. |
| `Apps/Web` | Browser workspace and HTTP API. |
| `Src/Runtime` | Configuration and dataset handling shared by training and inference. |
| `Src/IO` | JSON and checkpoint serialization. |
| `Models` | Model definitions. |
| `Configs/Training` | Model and tokenizer training configurations. |
| `Configs/Serving` | DeepSeek, Stable Diffusion, and FLUX.2 server configurations. |
| `Tests` | Numerical, model, and integration checks. |
