# Elman

Elman is a compact deep learning framework with CPU, CUDA, and Metal execution. It
provides tensor operations, a dynamic computation graph, reverse mode automatic
differentiation, model training, checkpoint loading, and inference.

## Capabilities

### Tensor types and operators

- Tensor operations cover elementwise arithmetic and comparisons, shape transforms,
  indexing and scatter/gather, reductions, matrix multiplication, convolution,
  pooling, normalization, losses, random sampling, and dropout.
- Specialized operators include attention, rotary embeddings, quantized affine
  transforms, and packed low-bit linear projections. Supported data types include
  floating-point, integer, Boolean, complex, and quantized formats; availability
  varies by operation and backend.

### Automatic differentiation

- Dynamic computation graphs provide reverse-mode automatic differentiation,
  gradient accumulation, and graph mutation checks.

### Execution backends

- CPU, NVIDIA CUDA, and Apple Metal are available through a common backend
  interface. CUDA supports execution across multiple devices, including layer placement
  tensor parallelism. Apple execution uses Metal and MPSGraph.

### Training

- Train and evaluate regression, classification, and language-model tasks through
  the native training API or JSON configurations. Declarative JSON tensor graphs
  can also be trained; PPO and policy/value components support reinforcement
  learning.
- The training loop supports SGD, Adam, and AdamW; gradient clipping and
  accumulation; constant or cosine learning-rate schedules with warmup; checkpoint
  save/resume; and FP32, FP16, or BF16 precision. FP16 training uses dynamic loss
  scaling.
- Multi-process data-parallel training uses TCP collectives.

### Inference

- Run model or tensor-graph inference, generate text with greedy or sampled
  decoding, synthesize images with diffusion, and transcribe audio with Whisper.
- Decoder inference supports KV caching and chunked prefill. Qwen2 serving adds
  continuous batching, paged KV storage, and prompt-prefix reuse; CUDA Qwen2 also
  supports captured decode graphs.
- Quantized weights and fused attention, normalization, and projection paths
  provide additional inference options where supported.

### Reinforcement learning

- PPO supports continuous-action actor-critic training with rollouts from pluggable
  environments, generalized advantage estimation, clipped policy updates, value
  loss, and minibatch optimization across multiple epochs.
- The RL toolkit also provides discounted returns and categorical policy-gradient
  losses for discrete-action policies, including legal-action masking.

### Model architectures

- Text generation: Qwen2 and DeepSeek V2/V3, including grouped-query attention,
  rotary positions, DeepSeek latent attention and sparse experts.
- Image generation: Stable Diffusion and FLUX.2.
- Speech recognition: Whisper.
- Others: AlphaZero-style spatial policy/value networks.

## Command line

Run these commands from the repository root with `elman` on your `PATH`.

```sh
elman --help
elman devices
```

### Serving

Set `model.path` in each serving configuration to the local checkpoint
directory. Relative paths resolve from the JSON file's directory. The
text and image examples select CUDA by default. Set `model.backend` to an available backend
reported by `elman devices`.

```sh
# DeepSeek chat and API at http://127.0.0.1:8080/
elman --server-config Configs/Serving/DeepSeekServer.json

# Stable Diffusion workspace and API at http://127.0.0.1:8081/
elman --server-config Configs/Serving/StableDiffusionServer.json

# FLUX.2 Klein workspace and API at http://127.0.0.1:8082/
elman --server-config Configs/Serving/Flux2Server.json
```

Run each model in a separate terminal. The FLUX.2 configuration uses the
Apache 2.0 checkpoint. Download its weights and supporting files with the
Hugging Face CLI before starting the server. [FLUX.2 Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)

```sh
hf download black-forest-labs/FLUX.2-klein-4B --revision e7b7dc27f91deacad38e78976d1f2b499d76a294 --local-dir Models/FLUX.2-klein-4B --include "LICENSE.md" --include "flux-2-klein-4b.safetensors" --include "text_encoder/*" --include "tokenizer/*" --include "vae/*"
```

### Training and evaluation

Set `data.path` in the training configuration to the dataset CSV before running
these commands. `infer` loads the checkpoint written by `train`.

```sh
elman train Configs/Training/LinearRegression.json
elman infer Configs/Training/LinearRegression.json
```

## Examples

### LLM

Elman runs DeepSeek-R1-Distill-Qwen-1.5B directly from its official weights and
streams the answer through the browser workspace. The serving path combines
tokenization, Qwen2 decoding, KV caching, and an HTTP chat API.

<img width="1280" height="720" alt="DeepSeek" src="https://github.com/user-attachments/assets/d2e1a5f8-2efc-498c-bde2-471042453396" />

### Image Generation

The browser workspace generates images with FLUX.2 Klein 4B using Elman's
flow-matching sampler. It shows denoising progress and previews before the final
image is ready.

<img width="1280" height="720" alt="Flux2" src="https://github.com/user-attachments/assets/e93f1477-32ec-4e70-a012-ab03e9471332" />

### Gaussian Splatting

Trains the Utah Teapot as a set of 3D Gaussians. Visualize in [EvansRenderer](https://github.com/atlantis13579/EvansRenderer).
EvansRenderer loads the trained model, projects and blends the splats, and renders a rotating
view of the teapot.

https://github.com/user-attachments/assets/5a8be4db-2606-49c4-b3b1-829cd3e29512

### Robotics Reinforcement Learning

Elman's policy and weights training drives the eight-legged robot in
[Riemann](https://github.com/atlantis13579/Riemann) Physics Playground. Riemann supplies the
articulated-body physics and visualizes the robot during PPO training.

https://github.com/user-attachments/assets/fc05cc57-90a3-454e-8f67-ed14fcc1db9a
