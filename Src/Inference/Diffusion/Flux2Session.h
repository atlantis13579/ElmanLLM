#pragma once

#include "Inference/Diffusion/StableDiffusion.h"
#include "Models/Diffusion/Flux2Text.h"
#include "Models/Diffusion/Flux2Transformer.h"
#include "Text/HuggingFaceTokenizer.h"

namespace elman {

class Flux2Session {
 public:
  Flux2Session(const std::filesystem::path& root, Backend backend, int device, DType dtype);
  StableDiffusionOutput generate(const DiffusionInput& input, const DiffusionTrace& trace = {},
                                 const DiffusionProgressCallback& progress = {},
                                 const DiffusionStepCallback& after_step = {}) const;
  Tensor decode_preview(const Tensor& latent, int max_side) const;

 private:
  Backend backend_;
  int device_;
  DType dtype_;
  HuggingFaceTokenizer tokenizer_;
  flux2::TextEncoder text_encoder_;
  flux2::Transformer transformer_;
  InferenceSession vae_decoder_;

  Tensor decode(const Tensor& latent) const;
};

}  // namespace elman
