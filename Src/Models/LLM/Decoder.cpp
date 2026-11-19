#include "Models/LLM/Decoder.h"
#include <stdexcept>

namespace elman {
DecoderOutput DecoderModel::forward_paged(const Tensor&, PagedKVCache&, const Tensor&, const Tensor&,
                                         bool, DecoderLogits, const DecoderTrace&) const {
    throw std::invalid_argument("This decoder requires per-layer contiguous caches; paged execution is unavailable");
}
DecoderOutput DecoderModel::forward_paged_batch(const Tensor&, PagedKVBatch&, const Tensor&, const Tensor&,
                                               bool, DecoderLogits, const DecoderTrace&) const {
    throw std::invalid_argument("This decoder does not implement paged request batching");
}
} // namespace elman
