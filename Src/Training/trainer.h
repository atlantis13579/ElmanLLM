#pragma once

#include <string>

#include "Models/Model.h"
#include "Runtime/RunOptions.h"

namespace elman {

class BatchDataset;
class ProcessGroup;

struct TrainingRun {
  Json definition;
  Json settings;
  Backend backend = Backend::CPU;
  uint32_t seed = 1;
  int steps = 0;
  std::string checkpoint, resume, metrics, graph;
};

int train_model(Model& model, const BatchDataset& data, const TrainingRun& run, ProcessGroup& group);
int train(const std::string& configuration, const RunOptions& options = {});

}  // namespace elman
