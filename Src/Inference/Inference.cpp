#include "Inference/Inference.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "IO/Checkpoint.h"
#include "Runtime/RunContext.h"

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

Json integer_array(const std::vector<int>& numbers) {
  Json::Array values;
  for (int number : numbers) {
    values.emplace_back(number);
  }
  return Json(std::move(values));
}

}  // namespace

int infer(const std::string& configuration, const RunOptions& options) {
  Configuration config(configuration, options);
  Dataset data(config);
  std::string checkpoint =
      options.checkpoint.empty()
          ? config.output("checkpoint", "../Intermediate/checkpoint.elman")
          : options.checkpoint;
  load_checkpoint(checkpoint, *config.model, config.definition);

  std::vector<size_t> order(data.inputs.size());
  std::iota(order.begin(), order.end(), 0);
  Json::Array predictions;
  size_t correct = 0;
  size_t count = 0;
  double total_loss = 0;
  for (size_t sample = 0; sample < order.size(); ++sample) {
    auto batch = data.batch(order, sample, 1, config.backend);
    auto logits = config.model->forward(batch.first, false, config.seed);
    auto values = logits.data();
    auto targets = batch.second.data();
    float loss =
        classification_loss(logits, batch.second, data.classes).data().at(0);
    require(std::isfinite(loss), "Non-finite inference loss");
    total_loss += loss;

    std::vector<int> predicted;
    std::vector<int> truth;
    for (size_t row = 0; row < targets.size(); ++row) {
      int best = 0;
      for (int cls = 0; cls < data.classes; ++cls) {
        require(std::isfinite(values[row * data.classes + cls]),
                "Non-finite inference output");
        if (values[row * data.classes + cls] >
            values[row * data.classes + best]) {
          best = cls;
        }
      }
      predicted.push_back(best);
      truth.push_back(static_cast<int>(targets[row]));
      if (best == truth.back()) {
        ++correct;
      }
      ++count;
    }
    predictions.emplace_back(
        Json::Object{{"sample", static_cast<double>(sample)},
                     {"predicted", integer_array(predicted)},
                     {"target", integer_array(truth)},
                     {"loss", static_cast<double>(loss)}});
  }

  Json result(Json::Object{{"format", "elman.inference"},
                           {"version", 1},
                           {"model", config.model->name()},
                           {"backend", backend_name(config.backend)},
                           {"loss", total_loss / order.size()},
                           {"accuracy", static_cast<double>(correct) / count},
                           {"predictions", std::move(predictions)}});
  std::cout << result.dump() << '\n';
  if (!options.output.empty()) {
    result.write(options.output);
  }
  return 0;
}

}  // namespace elman
