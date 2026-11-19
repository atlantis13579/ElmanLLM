#pragma once

#include "Models/Model.h"
#include "Training/LossScaler.h"
#include "Training/Sampler.h"

namespace elman {

struct TrainingState {
    uint64_t step = 0;
    std::string optimizer;
    std::vector<std::vector<float>> first_moment;
    std::vector<std::vector<float>> second_moment;
    // Native tensor slots retain Double or complex parameter dtype and backend.
    // The historical field names remain compatible with existing callers.
    std::vector<Tensor> first_moment_double;
    std::vector<Tensor> second_moment_double;
    // Low-format parameters retain Float32 tensor moments on their backend.
    // Legacy arithmetic dtypes continue to use the Float32 vectors above.
    std::vector<Tensor> first_moment_float;
    std::vector<Tensor> second_moment_float;
    // Low-format and complex SGD use this flag to distinguish an empty history from
    // a stored effective-gradient/momentum buffer, including an all-zero one.
    std::vector<uint8_t> sgd_buffer_initialized;
    bool resident_float32 = false;
    bool uses_resident_float32(const Tensor& tensor) const {
        return resident_float32 && tensor.dtype() == DType::Float32 && tensor.backend() == Backend::CUDA;
    }
};

struct TrainingProgress {
    SamplerState sampler;
    std::string configuration;
    uint64_t checksum = 0;
    uint64_t attempted_steps = 0;
    LossScaleState loss_scale;
};

void save_checkpoint(const std::string& path, const Model& model, const Json& definition,
                     const TrainingState& state, const TrainingProgress* progress = nullptr);
// If progress is provided, its expected dataset size bounds the sampler read.
// Old checkpoints return an empty configuration and sampler.
void load_checkpoint(const std::string& path, Model& model, const Json& definition,
                     TrainingState* state = nullptr, TrainingProgress* progress = nullptr,
                     size_t expected_samples = 0);

}
