#include "Inference/LLM/DevicePlacement.h"
#include "Core/Graph.h"
#include <algorithm>
#include <set>
#include <stdexcept>

namespace elman {
int decoder_layer_device(int layer, int layers, const std::vector<int>& ordinals) {
    if (layers <= 0 || layer < 0 || layer >= layers || ordinals.empty() || ordinals.size() > size_t(layers))
        throw std::invalid_argument("Invalid decoder layer placement");
    return ordinals[size_t(layer) * ordinals.size() / size_t(layers)];
}
DeviceStage::DeviceStage(Backend backend, int ordinal, const ExecutionStream* stream, bool asynchronous, bool pinned)
    : owner_(backend, ordinal) {
    if (stream) execution_ = std::make_unique<StreamScope>(*stream, asynchronous, pinned);
}
DecoderPlacement::DecoderPlacement(Backend backend, int layers, std::vector<int> ordinals, int simulated_devices)
    : backend_(backend), layers_(layers), stage_count_(0), simulated_(simulated_devices != 0), ordinals_(std::move(ordinals)) {
    if (layers <= 0) throw std::invalid_argument("Decoder placement needs positive layer count");
    if (ordinals_.empty()) ordinals_.push_back(current_device(backend));
    if (simulated_ && (backend != Backend::CUDA || ordinals_.size() != 1 || simulated_devices < 2 || simulated_devices > layers))
        throw std::invalid_argument("Device simulation requires one CUDA device and two or more nonempty layer stages");
    stage_count_ = simulated_ ? simulated_devices : int(ordinals_.size());
    if (ordinals_.size() > size_t(layers)) throw std::invalid_argument("More devices than decoder layers");
    std::set<int> unique;
    std::vector<DeviceInfo> available = devices(backend);
    for (int ordinal : ordinals_) {
        if (!unique.insert(ordinal).second) throw std::invalid_argument("Duplicate decoder device ordinal");
        if (std::none_of(available.begin(), available.end(), [&](const DeviceInfo& device) { return device.ordinal == ordinal; }))
            throw std::invalid_argument("Requested decoder device is unavailable: " + std::to_string(ordinal));
    }
    if (distributed()) {
        if (backend != Backend::CUDA) throw std::invalid_argument("Layer partitioning requires CUDA devices");
        for (int stage = 0; stage < stage_count_; ++stage) {
            const int ordinal = simulated_ ? ordinals_.front() : ordinals_[stage];
            DeviceScope owner(backend, ordinal);
            streams_.push_back(std::make_unique<ExecutionStream>(backend));
        }
    }
}
DecoderPlacement::DecoderPlacement(DecoderPlacement&& other) noexcept
    : backend_(other.backend_), layers_(other.layers_), stage_count_(other.stage_count_), simulated_(other.simulated_),
      ordinals_(std::move(other.ordinals_)), streams_(std::move(other.streams_)),
      activation_transfers_(other.activation_transfers_.load()) {}
int DecoderPlacement::layer_stage(int layer) const {
    if (layer < 0 || layer >= layers_) throw std::invalid_argument("Decoder layer is outside the placement");
    return int(size_t(layer) * stage_count_ / size_t(layers_));
}
int DecoderPlacement::layer_device(int layer) const {
    const int stage = layer_stage(layer);
    return simulated_ ? ordinals_.front() : ordinals_[stage];
}
DeviceStage DecoderPlacement::enter(int ordinal) const {
    const auto found = std::find(ordinals_.begin(), ordinals_.end(), ordinal);
    if (found == ordinals_.end()) throw std::invalid_argument("Device is outside the decoder placement");
    return enter_stage(int(found - ordinals_.begin()));
}
DeviceStage DecoderPlacement::enter_stage(int stage) const {
    if (stage < 0 || stage >= stage_count_) throw std::invalid_argument("Stage is outside the decoder placement");
    const int ordinal = simulated_ ? ordinals_.front() : ordinals_[stage];
    return DeviceStage(backend_, ordinal, streams_.empty() ? nullptr : streams_[stage].get(),
                       asynchronous_execution(backend_), pinned_staging_enabled(backend_));
}
DeviceStage DecoderPlacement::enter_layer(int layer) const { return enter_stage(layer_stage(layer)); }
Tensor DecoderPlacement::transfer_to_layer(const Tensor& hidden, int layer) const {
    const int ordinal = layer_device(layer);
    if (hidden.device() != ordinal) { ++activation_transfers_; return hidden.to_device(ordinal); }
    if (simulated_ && layer > 0 && layer_stage(layer) != layer_stage(layer - 1)) {
        ++activation_transfers_;
        // A real copy prevents the simulated stages from sharing the activation
        // allocation and exercises the same cross-queue read/write dependencies.
        Tensor copy = Tensor::zeros(hidden.shape(), backend_, false, hidden.dtype());
        copy_device_storage(copy.node()->value, hidden.node()->value);
        return copy;
    }
    return hidden;
}
void DecoderPlacement::synchronize() const {
    for (const std::unique_ptr<ExecutionStream>& stream : streams_) stream->synchronize();
}
const Tensor& DeviceTensorCopies::get(int ordinal) {
    if (!source_.defined() || source_.device() == ordinal) return source_;
    auto found = copies_.find(ordinal);
    if (found == copies_.end()) found = copies_.emplace(ordinal, source_.to_device(ordinal)).first;
    return found->second;
}
} // namespace elman
