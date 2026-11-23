#include "Inference/LLM/Sampling.h"
#include "Core/LocalFunction.h"
#include "Core/Graph.h"
#include "Core/Memory.h"
#include "Core/TensorDispatch.h"
#ifdef ELMAN_HAS_CUDA
#include "Backends/CudaGreedy.h"
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(std::string("Sampling: ") + message);
}
void matrix(const Tensor& x) {
    require(x.defined() && x.shape().size() == 2 && x.shape()[0] > 0 && x.shape()[1] > 0,
            "expected nonempty [batch, vocabulary] tensor");
    require(is_arithmetic_float(x.dtype()) || is_low_precision_float(x.dtype()), "expected an implemented floating input");
}
}
float SamplingRng::next() {
    uint64_t z = (state_ += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return float(z >> 40) * 0x1p-24f;
}
double SamplingRng::next_double() {
    uint64_t z = (state_ += UINT64_C(0x9e3779b97f4a7c15));
    z = (z ^ (z >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    z = (z ^ (z >> 27)) * UINT64_C(0x94d049bb133111eb);
    z ^= z >> 31;
    return double(z >> 11) * (1. / 9007199254740992.);
}
int32_t greedy_token(const Tensor& logits) {
    require(logits.defined() && logits.numel() != 0, "empty greedy vocabulary");
#ifdef ELMAN_HAS_CUDA
    if (logits.backend() == Backend::CUDA && logits.dtype() == DType::Float32 && logits.numel() > 256) {
        int32_t id = greedy_id_cuda(logits.node()->value);
        require(id >= 0, "model produced nonfinite logits");
        return id;
    }
#endif
    MemoryReservation host_logits(Backend::CPU, checked_memory_mul(logits.numel(), 8));
    std::vector<float> values = logits.data();
    auto best = values.begin();
    for (auto it = values.begin(); it != values.end(); ++it) {
        require(std::isfinite(*it), "model produced nonfinite logits");
        if (*it > *best) best = it;
    }
    return static_cast<int32_t>(best - values.begin());
}
void validate_sampling(const SamplingOptions& options) {
    require(std::isfinite(options.temperature) && options.temperature > 0, "temperature must be finite and positive");
    require(options.top_k >= 0, "top_k must be nonnegative");
    require(std::isfinite(options.top_p) && options.top_p >= 0 && options.top_p <= 1, "top_p must be in [0, 1]");
    require(std::isfinite(options.repetition_penalty) && options.repetition_penalty > 0,
            "repetition_penalty must be finite and positive");
}
Tensor sampling_scores(const Tensor& logits, const Tensor& history, const SamplingOptions& options) {
    validate_sampling(options);
    matrix(logits);
    int rows = logits.shape()[0], width = logits.shape()[1];
    require(width < std::numeric_limits<int>::max(), "vocabulary is too large");
    require(history.defined() && history.dtype() == DType::Int32 && history.backend() == logits.backend() &&
            history.shape().size() == 2 && history.shape()[0] == rows, "invalid history shape, dtype or backend");
    const bool wide = logits.dtype() == DType::Float64;
    const DType dtype = wide ? DType::Float64 : DType::Float32;
    LocalFunction scalar{[&](double value) { return wide ? Tensor::from_double_data({}, {value}, logits.backend()) :
                                                   Tensor::from_data({}, {float(value)}, logits.backend()); }};
    Tensor scores = cast(logits.detach(), dtype);
    if (history.shape()[1]) {
        // The extra column records invalid IDs without an out-of-bounds write.
        Tensor present = dispatch_tensor("token_presence", {rows, width + 1}, {history}, {}, DType::Int32);
        for (int invalid : slice(present, 1, width, width + 1).int_data())
            require(!invalid, "history ID outside vocabulary");
        if (options.repetition_penalty != 1) {
            Tensor zero = scalar(0.);
            double penalty = wide ? options.repetition_penalty : double(float(options.repetition_penalty));
            Tensor penalized = where(lt(scores, zero), mul(scores, penalty), div(scores, penalty));
            scores = where(cast(slice(present, 1, 0, width), DType::Bool), penalized, scores);
        }
    }
    if (options.temperature != 1) scores = div(scores, wide ? options.temperature : double(float(options.temperature)));
    if ((!options.top_k || options.top_k >= width) && options.top_p == 1) return scores;
    std::pair<Tensor, Tensor> ordered = sort(scores, 1);
    Tensor filtered = ordered.first;
    Tensor negative_infinity = scalar(-std::numeric_limits<double>::infinity());
    if (options.top_k && options.top_k < width) {
        Tensor cutoff = slice(filtered, 1, width - options.top_k, width - options.top_k + 1);
        filtered = where(lt(filtered, cutoff), negative_infinity, filtered);
    }
    if (options.top_p < 1) {
        Tensor cumulative = cumsum(softmax(filtered, 1), 1);
        Tensor cutoff = scalar(1 - options.top_p);
        Tensor masked = where(le(cumulative, cutoff), negative_infinity, filtered);
        // Keep at least one token, including top_p=0 and a one-token vocabulary.
        filtered = concat({slice(masked, 1, 0, width - 1), slice(filtered, 1, width - 1, width)}, 1);
    }
    return dispatch_tensor("sort_backward", scores.shape(), {filtered, ordered.second}, {1}, dtype);
}
std::vector<int32_t> categorical_sample(const Tensor& weights, const std::vector<float>& uniforms) {
    matrix(weights);
    if (weights.dtype() == DType::Float64) return categorical_sample_double(weights, std::vector<double>(uniforms.begin(), uniforms.end()));
    require(uniforms.size() == size_t(weights.shape()[0]), "one uniform is required per row");
    for (float value : uniforms) require(std::isfinite(value) && value >= 0 && value < 1, "uniform must be in [0, 1)");
    Tensor values = is_low_precision_float(weights.dtype()) ? cast(weights.detach(), DType::Float32) : weights.detach();
    Tensor cumulative = cumsum(values, 1);
    Tensor random = Tensor::from_data({weights.shape()[0]}, uniforms, weights.backend());
    std::vector<int32_t> ids = dispatch_tensor("categorical", {weights.shape()[0]}, {values, cumulative, random}, {}, DType::Int32).int_data();
    for (int id : ids) require(id >= 0, "invalid probability distribution");
    return ids;
}
std::vector<int32_t> categorical_sample_double(const Tensor& weights, const std::vector<double>& uniforms) {
    matrix(weights);
    require(weights.dtype() == DType::Float64, "Double uniforms require Double weights");
    require(uniforms.size() == size_t(weights.shape()[0]), "one uniform is required per row");
    for (double value : uniforms) require(std::isfinite(value) && value >= 0. && value < 1., "uniform must be in [0, 1)");
    Tensor values = weights.detach();
    Tensor cumulative = cumsum(values, 1);
    Tensor random = Tensor::from_double_data({weights.shape()[0]}, uniforms, weights.backend());
    std::vector<int32_t> ids = dispatch_tensor("categorical", {weights.shape()[0]}, {values, cumulative, random}, {}, DType::Int32).int_data();
    for (int id : ids) require(id >= 0, "invalid probability distribution");
    return ids;
}
std::vector<int32_t> sample_tokens(const Tensor& logits, const Tensor& history,
                                 const SamplingOptions& options, SamplingRng& rng) {
    Tensor probabilities = softmax(sampling_scores(logits, history, options), 1);
    if (probabilities.dtype() == DType::Float64) {
        std::vector<double> uniforms(probabilities.shape()[0]);
        for (double& value : uniforms) value = rng.next_double();
        return categorical_sample_double(probabilities, uniforms);
    }
    std::vector<float> uniforms(probabilities.shape()[0]);
    for (float& value : uniforms) value = rng.next();
    return categorical_sample(probabilities, uniforms);
}
} // namespace elman
