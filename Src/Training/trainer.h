#pragma once

#include <string>

#include "Models/Model.h"

namespace elman {

class BatchDataset;
class ProcessGroup;

struct TrainingRun {
  Json definition;
  Json settings;
  Backend backend = Backend::CPU;
  uint32_t seed = 1;
  int steps = 0;
  std::string checkpoint, resume, metrics, graph, best_checkpoint;
};

int train_model(Model& model, const BatchDataset& data, const TrainingRun& run, ProcessGroup& group,
                const BatchDataset* validation = nullptr);

}  // namespace elman
