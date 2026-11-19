#pragma once

#include "IO/ComponentWeights.h"
#include "Inference/InferenceSession.h"

namespace elman::flux2 {
// Load either half independently so reference encoding can release its weights
// before denoising. Images use NCHW RGB in [-1,1], latents use [B,128,H,W].
InferenceSession load_vae_encoder(const io::ComponentWeights& weights);
InferenceSession load_vae_decoder(const io::ComponentWeights& weights);
}  // namespace elman::flux2
