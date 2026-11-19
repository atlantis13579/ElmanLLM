#include "Inference/Diffusion/Flux2Session.h"

#include <algorithm>
#include <stdexcept>

#include "Core/Device.h"
#include "Core/Graph.h"
#include "Inference/Diffusion/Flux2Sampler.h"
#include "Models/Diffusion/Flux2Vae.h"

namespace elman {
namespace {

constexpr int spatial_factor = 16;
constexpr int text_length = 512;

InferenceSession load_decoder(const std::filesystem::path& root, Backend backend, int device, DType dtype) {
  DeviceScope owner(backend, device);
  return flux2::load_vae_decoder(io::ComponentWeights(root / "vae", "diffusion_pytorch_model", backend, dtype));
}

}  // namespace

Flux2Session::Flux2Session(const std::filesystem::path& root, Backend backend, int device, DType dtype)
    : backend_(backend), device_(device), dtype_(dtype),
      tokenizer_((root / "tokenizer").u8string()), text_encoder_(root / "text_encoder", dtype),
      // Keep the transformer weights in host memory and transfer each projection as needed.
      transformer_(io::ComponentWeights(root, "flux-2-klein-4b", Backend::CPU, dtype), flux2::Variant::Klein4B),
      vae_decoder_(load_decoder(root, backend, device, dtype)) {
  if (tokenizer_.pad_id() < 0) throw std::invalid_argument("FLUX.2 Klein needs a tokenizer padding token");
}

StableDiffusionOutput Flux2Session::generate(const DiffusionInput& input, const DiffusionTrace& trace,
                                            const DiffusionProgressCallback& progress,
                                            const DiffusionStepCallback& after_step) const {
  if (input.width <= 0 || input.height <= 0 || input.width % spatial_factor || input.height % spatial_factor ||
      input.steps < 1 || input.guidance != 1 || !input.negative_prompt.empty())
    throw std::invalid_argument("FLUX.2 Klein requires dimensions divisible by 16, guidance 1 and no negative prompt");
  DeviceScope owner(backend_, device_);
  graph::RecordingGuard no_grad;
  using Stage = DiffusionProgress::Stage;
  auto update = [&](Stage stage, int step) {
    if (progress) progress({stage, step, input.steps});
  };

  update(Stage::Preparing, 0);
  const Shape shape{1, 128, input.height / spatial_factor, input.width / spatial_factor};
  auto noise = input.initial_latent;
  if (!noise.defined()) {
    noise = Tensor::normal(shape, 0, 1, input.seed, backend_, false, dtype_);
  } else if (noise.shape() != shape || noise.backend() != backend_ || noise.device() != device_ || noise.dtype() != dtype_) {
    throw std::invalid_argument("FLUX.2 initial latent has an incompatible device, dtype or shape");
  }
  if (trace) trace("initial_latent", noise);

  update(Stage::EncodingPrompt, 0);
  auto tokens = tokenizer_.encode("<|im_start|>user\n" + input.prompt +
                                  "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", false);
  const auto valid = std::min(tokens.size(), size_t(text_length));
  tokens.resize(text_length, tokenizer_.pad_id());
  std::vector<int32_t> mask(text_length, 0);
  std::fill_n(mask.begin(), valid, 1);
  auto prompt = text_encoder_.encode(Tensor::from_int_data({1, text_length}, tokens, backend_),
                                     Tensor::from_int_data({1, text_length}, mask, backend_));
  if (trace) trace("text_embeddings", prompt);

  flux2::SamplingOptions options;
  options.steps = input.steps;
  options.before_step = [&](int step) { update(Stage::Denoising, step); };
  if (after_step) {
    options.on_step = [&](int step, const Tensor& latent) { after_step(step, input.steps, latent); };
  }
  auto latent = flux2::sample(transformer_, noise, prompt, {}, {}, options);
  update(Stage::Decoding, input.steps);
  auto image = decode(latent);
  return {std::move(latent), std::move(image)};
}

Tensor Flux2Session::decode(const Tensor& latent) const {
  auto image = add(mul(vae_decoder_.run({{"latent", latent}}).at("image"), 0.5), 0.5);
  return maximum(Tensor::zeros({}, backend_, false, dtype_), minimum(image, Tensor::ones({}, backend_, false, dtype_)));
}

Tensor Flux2Session::decode_preview(const Tensor& latent, int max_side) const {
  DeviceScope owner(backend_, device_);
  graph::RecordingGuard no_grad;
  return decode(diffusion_preview_latent(latent, max_side / spatial_factor));
}

}  // namespace elman
