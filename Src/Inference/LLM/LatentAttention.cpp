#include "Inference/LLM/LatentAttention.h"
#include "Core/Graph.h"
#include "Core/Device.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(std::string("Latent attention: ") + message);
}
void tensor(const Tensor& value, Backend backend, DType dtype) {
    require(value.defined() && value.backend() == backend && value.dtype() == dtype && !value.requires_grad(),
            "operand backend, dtype or inference state differs");
}
}
int LatentKVCache::length() const {
    if (!latent.defined()) {
        require(!rotary.defined(), "partial compressed cache");
        return 0;
    }
    require(latent.shape().size() == 3 && rotary.defined() && rotary.shape().size() == 3 &&
            latent.shape()[0] == rotary.shape()[0] && latent.shape()[1] == rotary.shape()[1], "cache geometry differs");
    return latent.shape()[1];
}
LatentKVCache append_latent_cache(const LatentKVCache* previous, const Tensor& latent, const Tensor& rotary) {
    require(latent.defined() && rotary.defined() && latent.shape().size() == 3 && rotary.shape().size() == 3 &&
            is_compute_float(latent.dtype()), "invalid compressed cache operands");
    tensor(latent, latent.backend(), latent.dtype());
    tensor(rotary, latent.backend(), latent.dtype());
    require(latent.shape()[0] > 0 && latent.shape()[2] > 0 && rotary.shape()[2] > 0 &&
            rotary.shape()[0] == latent.shape()[0] && rotary.shape()[1] == latent.shape()[1], "cache geometry differs");
    if (!previous || !previous->latent.defined()) {
        require(!previous || !previous->rotary.defined(), "partial previous cache");
        return {latent, rotary};
    }
    tensor(previous->latent, latent.backend(), latent.dtype());
    tensor(previous->rotary, latent.backend(), latent.dtype());
    require(previous->latent.shape().size() == 3 && previous->rotary.shape().size() == 3 &&
            previous->latent.shape()[0] == latent.shape()[0] && previous->latent.shape()[2] == latent.shape()[2] &&
            previous->rotary.shape()[0] == rotary.shape()[0] && previous->rotary.shape()[2] == rotary.shape()[2] &&
            previous->latent.shape()[1] == previous->rotary.shape()[1], "previous cache geometry differs");
    return {concat({previous->latent, latent}, 1), concat({previous->rotary, rotary}, 1)};
}

Tensor latent_attention(const Tensor& query_plain, const Tensor& query_rotary, const LatentKVCache& cache,
                        const Tensor& key_weight, const Tensor& value_weight, const Tensor& padding,
                        double scale, int causal_offset, int query_tile) {
    require(query_plain.defined() && query_plain.shape().size() == 4 && is_compute_float(query_plain.dtype()), "invalid query");
    DeviceScope owner(query_plain.backend(), query_plain.device());
    const Backend backend = query_plain.backend();
    const DType dtype = query_plain.dtype();
    for (const Tensor* value : {&query_plain, &query_rotary, &cache.latent, &cache.rotary, &key_weight, &value_weight}) {
        tensor(*value, backend, dtype);
        require(value->device() == query_plain.device(), "operand device differs");
    }
    const Shape& q = query_plain.shape();
    require(q[0] > 0 && q[1] > 0 && q[3] > 0 && query_rotary.shape().size() == 4 &&
            query_rotary.shape()[0] == q[0] && query_rotary.shape()[1] == q[1] && query_rotary.shape()[2] == q[2],
            "query geometry differs");
    require(cache.latent.shape().size() == 3 && cache.rotary.shape().size() == 3 &&
            cache.latent.shape()[0] == q[0] && cache.rotary.shape()[0] == q[0] &&
            cache.latent.shape()[1] == cache.rotary.shape()[1] && cache.rotary.shape()[2] == query_rotary.shape()[3] &&
            cache.latent.shape()[2] > 0 && cache.rotary.shape()[2] > 0, "compressed cache geometry differs");
    const int rank = cache.latent.shape()[2], keys = cache.latent.shape()[1];
    require(key_weight.shape() == Shape{q[1], q[3], rank} && value_weight.shape().size() == 3 &&
            value_weight.shape()[0] == q[1] && value_weight.shape()[2] == rank && value_weight.shape()[1] > 0,
            "absorbed projection geometry differs");
    require(std::isfinite(scale) && scale > 0 && causal_offset >= 0 && query_tile > 0 && query_tile <= 1024 &&
            int64_t(causal_offset) + q[2] <= keys, "invalid scale, tiling or causal position");
    if (padding.defined()) {
        tensor(padding, backend, dtype);
        require(padding.device() == query_plain.device(), "padding device differs");
        require(padding.shape() == Shape{q[0], 1, 1, keys}, "padding must have shape [batch,1,1,keys]");
    }
    graph::RecordingGuard inference;
    if (!q[2]) return Tensor::zeros({q[0], q[1], 0, value_weight.shape()[1]}, backend, false, dtype);
    Tensor latent = unsqueeze(cache.latent, 1), rotary = unsqueeze(cache.rotary, 1);
    const Tensor padding_mask = padding.defined() ? padding : Tensor::zeros({q[0], 1, 1, keys}, backend, false, dtype);
    std::vector<Tensor> tiles;
    for (int start = 0; start < q[2]; start += query_tile) {
        const int end = std::min(start + query_tile, q[2]);
        const bool full_query = start == 0 && end == q[2];
        Tensor compressed_query = matmul(full_query ? query_plain : slice(query_plain, 2, start, end), key_weight);
        Tensor scores = add(matmul_transposed(compressed_query, latent),
                          matmul_transposed(full_query ? query_rotary : slice(query_rotary, 2, start, end), rotary));
        scores = mul(scores, scale);
        Tensor mask = padding_mask;
        if (causal_offset + start < keys - 1) {
            Tensor future = cast(triu(Tensor::ones({1, 1, end - start, keys}, backend), causal_offset + start + 1), DType::Bool);
            const float minimum = dtype == DType::Float16 ? -65504.f :
                dtype == DType::BFloat16 ? -3.3895313892515355e38f : std::numeric_limits<float>::lowest();
            Tensor excluded = add(Tensor::zeros({1, 1, end - start, keys}, backend, false, dtype), minimum);
            mask = where(future, excluded, mask);
        }
        Tensor probabilities = attention_softmax(scores, mask);
        Tensor values = matmul(probabilities, latent);
        tiles.push_back(matmul_transposed(values, value_weight));
    }
    return tiles.size() == 1 ? tiles.front() : concat(tiles, 2);
}
} // namespace elman
