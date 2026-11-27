#pragma once

#include "Core/Tensor.h"

namespace elman::flux2 {

// Inference kernals accept Float32, Float16 and BFloat16 activations.
// Supply [B,L,4] coordinates in (time, height, width, text) order. The FP32
// [B,L,128] table can be shared by every block, and batch 1 broadcasts.
Tensor rotary_table(const Tensor& ids);
Tensor apply_rotary(const Tensor& heads, const Tensor& table);

// Posterior moments [B,64,2H,2W] or their mean [B,32,2H,2W] to [B,128,H,W].
// Statistics are checkpoint running statistics, indexed by (channel,dy,dx).
Tensor pack_latents(const Tensor& posterior, const Tensor& mean, const Tensor& variance);
Tensor unpack_latents(const Tensor& packed, const Tensor& mean, const Tensor& variance);

// Affine-free LayerNorm over D, followed by (1+scale)*normalized+shift.
// Modulation tensors have shape [B,1,D] or [B,L,D].
Tensor modulate(const Tensor& x, const Tensor& shift, const Tensor& scale);
Tensor timestep_embedding(const Tensor& times);
Tensor attention(const Tensor& query, const Tensor& key, const Tensor& value);

namespace detail {
void validate_kernel(const std::string& op, const BufferPtr& output, const std::vector<BufferPtr>& inputs,
                     const KernelParams& params);
}

}  // namespace elman::flux2
