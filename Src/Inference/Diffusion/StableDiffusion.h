#pragma once

#include "Inference/Diffusion/Ddim.h"
#include "Inference/InferenceSession.h"
#include "Text/ClipTokenizer.h"

namespace elman {

using DiffusionTrace = std::function<void(const std::string&, const Tensor&)>;

struct DiffusionProgress {
  enum class Stage { Preparing, EncodingPrompt, Denoising, Decoding };
  Stage stage;
  int step, total_steps;
};

using DiffusionProgressCallback = std::function<void(const DiffusionProgress&)>;
using DiffusionStepCallback = std::function<void(int completed, int total, const Tensor& predicted_original)>;
std::string diffusion_progress_text(const DiffusionProgress& progress);
Tensor diffusion_preview_latent(const Tensor& latent, int max_extent);

struct StableDiffusionOutput {
  Tensor latent;
  Tensor image;  // NCHW floating point in [0,1].
};

// The model components use Diffusers tensor layouts. The caller can provide
// an explicit initial latent to reproduce a reference denoising trajectory.
class StableDiffusion {
 public:
  StableDiffusion(InferenceSession text_encoder, InferenceSession denoiser,
                  InferenceSession vae_decoder, double latent_scaling);
  StableDiffusionOutput generate(const Tensor& prompt_ids, const Tensor& negative_ids,
                                 const Tensor& initial_latent, const DdimScheduler& scheduler,
                                 double guidance = 7.5, const DiffusionTrace& trace = {},
                                 const DiffusionProgressCallback& progress = {},
                                 const DiffusionStepCallback& after_step = {}) const;
  Tensor decode_latent(const Tensor& latent, const DiffusionTrace& trace = {}) const;

 private:
  InferenceSession text_encoder_, denoiser_, vae_decoder_;
  double latent_scaling_;
};

struct DiffusionInput {
  std::string prompt, negative_prompt;
  int width = 0, height = 0, steps = 20;
  uint32_t seed = 1;
  double guidance = 7.5;
  Tensor initial_latent;
};

class DiffusionSession {
 public:
  static DiffusionSession load(const std::filesystem::path& root, Backend backend, int device, DType dtype);
  StableDiffusionOutput generate(const DiffusionInput& input, const DiffusionTrace& trace = {},
                                 const DiffusionProgressCallback& progress = {},
                                 const DiffusionStepCallback& after_step = {}) const;
  Tensor decode_preview(const Tensor& latent, int max_side) const;

 private:
  DiffusionSession(StableDiffusion pipeline, ClipTokenizer tokenizer, Json scheduler,
                   Backend backend, int device, DType dtype, int channels, int spatial_factor);
  StableDiffusion pipeline_;
  ClipTokenizer tokenizer_;
  Json scheduler_;
  Backend backend_;
  int device_;
  DType dtype_;
  int channels_, spatial_factor_;
};

}  // namespace elman
