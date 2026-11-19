#pragma once

#include "Core/Device.h"
#include "Core/Tensor.h"
#include <map>
#include <atomic>

namespace elman {
int decoder_layer_device(int layer, int layers, const std::vector<int>& ordinals);

class DeviceStage {
public:
    DeviceStage(Backend backend, int ordinal, const ExecutionStream* stream, bool asynchronous, bool pinned);
private:
    DeviceScope owner_;
    std::unique_ptr<StreamScope> execution_;
};

// Contiguous layer partitioning. Embedding belongs to the first device and the
// final norm/head to the last. Each multi-device stage owns a nonblocking queue.
class DecoderPlacement {
public:
    DecoderPlacement(Backend backend, int layers, std::vector<int> ordinals = {}, int simulated_devices = 0);
    DecoderPlacement(DecoderPlacement&& other) noexcept;
    const std::vector<int>& ordinals() const { return ordinals_; }
    bool distributed() const { return stage_count_ > 1; }
    int stage_count() const { return stage_count_; }
    bool simulated() const { return simulated_; }
    size_t activation_transfers() const { return activation_transfers_.load(); }
    int input_device() const { return ordinals_.front(); }
    int output_device() const { return ordinals_.back(); }
    int layer_device(int layer) const;
    DeviceStage enter(int ordinal) const;
    DeviceStage enter_input() const { return enter_stage(0); }
    DeviceStage enter_output() const { return enter_stage(stage_count_ - 1); }
    DeviceStage enter_layer(int layer) const;
    Tensor transfer_to_layer(const Tensor& hidden, int layer) const;
    void synchronize() const;
private:
    Backend backend_;
    int layers_;
    int stage_count_;
    bool simulated_;
    std::vector<int> ordinals_;
    std::vector<std::unique_ptr<ExecutionStream>> streams_;
    mutable std::atomic<size_t> activation_transfers_{0};
    int layer_stage(int layer) const;
    DeviceStage enter_stage(int stage) const;
};

// Forward-local copies of masks/rotary tables: one transfer per device.
class DeviceTensorCopies {
public:
    explicit DeviceTensorCopies(Tensor source) : source_(std::move(source)) {}
    const Tensor& get(int ordinal);
private:
    Tensor source_;
    std::map<int, Tensor> copies_;
};
} // namespace elman
