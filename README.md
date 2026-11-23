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
- An HTTP API and a standalone browser workspace for streaming chat and image
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

The FLUX.2 web workspace defaults to 512 × 512 and four denoising steps.
The `POST /sdapi/v1/txt2img` endpoint also accepts a prompt and optional
width, height, steps, and seed. The image dimensions must be divisible by 16.

### Policy-value serving

Start a policy and value model with `elman --server-config PATH`, using a model definition and service configuration supplied by the client project.
The client connects directly over TCP for prediction, optimizer updates and checkpoint requests.
The client owns observations, trajectories, targets and the checkpoint schedule. [Protocol and training records](Docs/PolicyValue.md)

### Browser workspace

The HTTP serving examples set `http.web_root` to the repository's `Web`
directory. Open the model server's URL to use the browser workspace. Pass
`--open` to launch your default browser after the model and listener are ready.

```sh
elman --server-config Configs/Serving/Flux2Server.json --open
```

Elman serves static files directly from disk on each request. Edit the
frontend and refresh the browser while the server is running. For a custom
frontend, set `http.web_root` to its static output directory, with
`index.html` as the entry point. Relative paths resolve from the serving
configuration. Omit `http.web_root` for API-only serving.

The workspace uses the current origin for API requests. Set `?api=http://127.0.0.1:8082`
when serving it from an independent frontend server without a proxy.
Add that frontend's exact origin, including its scheme and port, to
`http.cors_origins`. The examples allow `http://127.0.0.1:8000`.

| Endpoint | Frontend integration |
| --- | --- |
| `GET /health` | Get the model name, type, and supported features. |
| `POST /v1/chat/completions` | Send JSON with `model`, `messages`, and `stream: true`. The SSE stream sends text in `choices[0].delta.content` and ends with `[DONE]`. |
| `POST /sdapi/v1/txt2img` | Send JSON with a `prompt` and use `Accept: text/event-stream` for `progress`, `preview`, and `result` events. Set `live_preview: true` to request previews. Preview images and final `result.images` are base64-encoded BMPs. |

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

### FLUX.2 Image Generation

The browser workspace generates images with FLUX.2 Klein 4B using Elman's
flow-matching sampler. It shows denoising progress and previews before the final
image is ready.

### Gaussian Splatting with EvansRenderer

Elman trains the Utah Teapot as a set of 3D Gaussians. [EvansRenderer](https://github.com/atlantis13579/EvansRenderer)
loads the trained model, projects and blends the splats, and renders a rotating
view of the teapot.

### Robotics Reinforcement Learning with Riemann Physics

Elman's policy and value training drives the eight-legged robot in
[Riemann Physics](https://github.com/atlantis13579/Riemann). Riemann Physics supplies the
articulated-body physics and visualizes the robot during PPO training.
