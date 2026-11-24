#include "Commands/Commands.h"
#include "Training/Trainer.h"
#include "Distributed/ProcessGroup.h"
#include "Runtime/RunContext.h"
#include "Core/Device.h"

namespace elman {
int train(const std::string& configuration, const RunOptions& options) {
  const ProcessGroupOptions distributed = process_group_environment();
  ProcessGroup group(distributed);
  const int cuda_device = group.size() > 1 ? distributed.local_rank : -1;
  Configuration config(configuration, options, cuda_device);
  DeviceScope device(config.backend, config.backend == Backend::CUDA && cuda_device >= 0
                                         ? cuda_device : current_device(config.backend));
  Dataset data(config);
  TrainingRun run;
  run.definition = config.definition;
  run.settings = config.json.at("training");
  run.backend = config.backend;
  run.seed = config.seed;
  run.steps = options.steps;
  run.checkpoint = options.checkpoint.empty()
      ? config.output("checkpoint", "../Intermediate/checkpoint.elman") : options.checkpoint;
  run.metrics = options.output.empty()
      ? config.output("metrics", "../Intermediate/metrics.json") : options.output;
  run.resume = options.resume;
  if (config.json.contains("output") && config.json.at("output").contains("graph"))
    run.graph = config.output("graph", "../Intermediate/graph.json");
  return train_model(*config.model, data, run, group);
}

}
