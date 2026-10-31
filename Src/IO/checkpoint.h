#pragma once

#include "Models/model.h"

namespace elman {

struct TrainingState {
    uint64_t step = 0;
    std::string optimizer;
    std::vector<std::vector<float>> first_moment;
    std::vector<std::vector<float>> second_moment;
};

void save_checkpoint(const std::string& path, const Model& model, const Json& definition, const TrainingState& state);
void load_checkpoint(const std::string& path, Model& model, const Json& definition, TrainingState* state = nullptr);

}
