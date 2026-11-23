#include "Inference/Diffusion/StableDiffusion.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "Core/Device.h"
#include "Core/Graph.h"
#include "Models/ClipText.h"
#include "Models/Diffusion/UNet.h"
#include "Models/Diffusion/Vae.h"

namespace elman {

std::string diffusion_progress_text(const DiffusionProgress& progress) {
  switch (progress.stage) {
    case DiffusionProgress::Stage::Preparing: return "Preparing latent";
    case DiffusionProgress::Stage::EncodingPrompt: return "Encoding prompt";
    case DiffusionProgress::Stage::Denoising:
      return "Denoising step " + std::to_string(progress.step) + "/" + std::to_string(progress.total_steps);
    case DiffusionProgress::Stage::Decoding: return "Decoding image (VAE)";
  }
  return {};
}

StableDiffusion::StableDiffusion(InferenceSession text_encoder, InferenceSession denoiser,
                                 InferenceSession vae_decoder, double latent_scaling)
    : text_encoder_(std::move(text_encoder)), denoiser_(std::move(denoiser)),
      vae_decoder_(std::move(vae_decoder)), latent_scaling_(latent_scaling) {
  if (!(latent_scaling_ > 0) || !std::isfinite(latent_scaling_))
    throw std::invalid_argument("Invalid Stable Diffusion latent scaling");
  for (const InferenceSession* component : {&text_encoder_, &vae_decoder_}) {
    if (component->backend() != denoiser_.backend() || component->device() != denoiser_.device())
      throw std::invalid_argument("Stable Diffusion sessions must use the same device");
  }
}

StableDiffusionOutput StableDiffusion::generate(const Tensor& prompt_ids, const Tensor& negative_ids,
                                                const Tensor& initial_latent, const DdimScheduler& scheduler,
                                                double guidance, const DiffusionTrace& trace,
                                                const DiffusionProgressCallback& progress,
                                                const DiffusionStepCallback& after_step) const {
  if (!std::isfinite(guidance) || guidance < 0 || initial_latent.shape().size() != 4 ||
      prompt_ids.shape().size() != 2 || initial_latent.shape()[0] != prompt_ids.shape()[0])
    throw std::invalid_argument("Invalid Stable Diffusion batch or guidance");
  const bool guided = guidance > 1.0;
  if (guided && (!negative_ids.defined() || negative_ids.shape() != prompt_ids.shape()))
    throw std::invalid_argument("Guided diffusion requires matching negative prompt input IDs");
  const Backend backend = denoiser_.backend();
  DeviceScope owner(backend, denoiser_.device());
  graph::RecordingGuard no_grad;
  const int steps = static_cast<int>(scheduler.timesteps().size());
  if (progress) progress({DiffusionProgress::Stage::EncodingPrompt, 0, steps});

  Tensor positive = text_encoder_.run({{"input_ids", prompt_ids}}).at("last_hidden_state");
  if (trace) trace("text_embeddings", positive);
  Tensor context = positive;
  if (guided) {
    Tensor negative = text_encoder_.run({{"input_ids", negative_ids}}).at("last_hidden_state");
    if (trace) trace("negative_text_embeddings", negative);
    context = concat({negative, positive}, 0);
  }
  Tensor latent = initial_latent.detach();
  if (trace) trace("initial_latent", latent);
  for (size_t step = 0; step < scheduler.timesteps().size(); ++step) {
    if (progress) progress({DiffusionProgress::Stage::Denoising, static_cast<int>(step + 1), steps});
    Tensor model_input = guided ? concat({latent, latent}, 0) : latent;
    Tensor timestep = Tensor::from_int_data({}, {scheduler.timesteps()[step]}, backend);
    Tensor noise = denoiser_.run({{"sample", model_input}, {"timestep", timestep},
                                 {"encoder_hidden_states", context}}).at("sample");
    if (guided) {
      const int batch = latent.shape()[0];
      Tensor unconditioned = slice(noise, 0, 0, batch);
      Tensor conditioned = slice(noise, 0, batch, batch * 2);
      noise = add(unconditioned, mul(sub(conditioned, unconditioned), guidance));
    }
    if (trace) trace("noise." + std::to_string(step), noise);
    DdimStep denoised = scheduler.step(step, noise, latent);
    latent = std::move(denoised.latent);
    if (trace) trace("latent." + std::to_string(step), latent);
    if (after_step) after_step(static_cast<int>(step + 1), steps, denoised.predicted_original);
  }
  if (progress) progress({DiffusionProgress::Stage::Decoding, steps, steps});
  Tensor image = decode_latent(latent, trace);
  return {std::move(latent), std::move(image)};
}

Tensor StableDiffusion::decode_latent(const Tensor& latent, const DiffusionTrace& trace) const {
  const Backend backend = vae_decoder_.backend();
  DeviceScope owner(backend, vae_decoder_.device());
  graph::RecordingGuard no_grad;
  Tensor decoded = vae_decoder_.run({{"latent", div(latent, latent_scaling_)}}).at("sample");
  if (trace) trace("vae_output", decoded);
  Tensor image = add(mul(decoded, 0.5), 0.5);
  return maximum(Tensor::zeros({}, backend, false, image.dtype()),
                 minimum(image, Tensor::ones({}, backend, false, image.dtype())));
}

DiffusionSession::DiffusionSession(StableDiffusion pipeline, ClipTokenizer tokenizer, Json scheduler,
                                   Backend backend, int device, DType dtype, int channels, int spatial_factor)
    : pipeline_(std::move(pipeline)), tokenizer_(std::move(tokenizer)), scheduler_(std::move(scheduler)),
      backend_(backend), device_(device), dtype_(dtype), channels_(channels), spatial_factor_(spatial_factor) {}

DiffusionSession DiffusionSession::load(const std::filesystem::path& root, Backend backend, int device, DType dtype) {
  if (!is_compute_float(dtype)) throw std::invalid_argument("Diffusion requires float32, float16 or bfloat16");
  DeviceScope owner(backend, device);
  Json scheduler = Json::read(root / "scheduler" / "scheduler_config.json");
  ClipTokenizer tokenizer((root / "tokenizer").string());
  InferenceSession text = load_clip_text_encoder(root / "text_encoder", backend, device, dtype);
  if (text.inputs().at("input_ids").shape[1] != tokenizer.length())
    throw std::invalid_argument("CLIP tokenizer and encoder length differ");
  InferenceSession denoiser = load_diffusion_unet(root / "unet", backend, device, dtype);
  VaeDecoder vae = load_diffusion_vae(root / "vae", backend, device, dtype);
  const int channels = denoiser.inputs().at("sample").shape[1];
  if (channels != vae.session.inputs().at("latent").shape[1] ||
      text.outputs().at("last_hidden_state").shape[2] != denoiser.inputs().at("encoder_hidden_states").shape[2])
    throw std::invalid_argument("Diffusion component dimensions are incompatible");
  StableDiffusion pipeline(std::move(text), std::move(denoiser), std::move(vae.session), vae.latent_scaling);
  return DiffusionSession(std::move(pipeline), std::move(tokenizer), std::move(scheduler),
                          backend, device, dtype, channels, vae.spatial_factor);
}

StableDiffusionOutput DiffusionSession::generate(const DiffusionInput& input, const DiffusionTrace& trace,
                                                 const DiffusionProgressCallback& progress,
                                                 const DiffusionStepCallback& after_step) const {
  DeviceScope owner(backend_, device_);
  if (input.width < 0 || input.height < 0 || !std::isfinite(input.guidance) || input.guidance < 0)
    throw std::invalid_argument("Invalid Diffusion dimensions or guidance");
  const DdimScheduler schedule(scheduler_, input.steps);
  if (progress) progress({DiffusionProgress::Stage::Preparing, 0, static_cast<int>(schedule.timesteps().size())});
  Tensor initial = input.initial_latent;
  if (initial.defined()) {
    const Shape& shape = initial.shape();
    if (initial.dtype() != dtype_ || initial.backend() != backend_ || initial.device() != device_ ||
        shape.size() != 4 || shape[0] != 1 || shape[1] != channels_ ||
        (input.width && int64_t(shape[3]) * spatial_factor_ != input.width) ||
        (input.height && int64_t(shape[2]) * spatial_factor_ != input.height))
      throw std::invalid_argument("Initial latent has an incompatible device, dtype or shape");
  } else {
    const int width = input.width ? input.width : 512;
    const int height = input.height ? input.height : 512;
    if (width % spatial_factor_ || height % spatial_factor_)
      throw std::invalid_argument("Image dimensions must be divisible by the VAE factor");
    initial = Tensor::normal({1, channels_, height / spatial_factor_, width / spatial_factor_},
                             0, 1, input.seed, backend_, false, dtype_);
  }
  LocalFunction ids{[&](const std::string& prompt) {
    return Tensor::from_int_data({1, tokenizer_.length()}, tokenizer_.encode(prompt), backend_);
  }};
  return pipeline_.generate(ids(input.prompt), input.guidance > 1 ? ids(input.negative_prompt) : Tensor{},
                            initial, schedule, input.guidance, trace, progress, after_step);
}

Tensor diffusion_preview_latent(const Tensor& latent, int max_extent) {
  const Shape& shape = latent.shape();
  if (shape.size() != 4 || shape[2] <= 0 || shape[3] <= 0 || max_extent < 1)
    throw std::invalid_argument("Invalid diffusion preview dimensions");
  const int extent = std::max(shape[2], shape[3]);
  if (extent <= max_extent) return latent;

  DeviceScope owner(latent.backend(), latent.device());
  graph::RecordingGuard no_grad;
  // Sample pixel centers in latent space before running the VAE.
  LocalFunction indices{[&](int size) {
    const int count = std::max(1, int((int64_t(size) * max_extent + extent / 2) / extent));
    std::vector<int32_t> selected;
    selected.reserve(count);
    for (int i = 0; i < count; ++i) selected.push_back(int32_t((2LL * i + 1) * size / (2LL * count)));
    return Tensor::from_int_data({count}, selected, latent.backend());
  }};
  return index_select(index_select(latent, 2, indices(shape[2])), 3, indices(shape[3]));
}

Tensor DiffusionSession::decode_preview(const Tensor& latent, int max_side) const {
  return pipeline_.decode_latent(diffusion_preview_latent(latent, max_side / spatial_factor_));
}

}  // namespace elman
