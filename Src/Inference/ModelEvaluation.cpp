#include "Inference/ModelEvaluation.h"

#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Inference/InferenceSession.h"
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
  load_inference_parameters(std::filesystem::u8path(checkpoint), *config.model, config.definition, config.backend);

  std::vector<size_t> order(data.inputs.size());
  std::iota(order.begin(), order.end(), 0);
  Json::Array predictions;
  size_t correct = 0;
  size_t count = 0;
  double total_loss = 0;
  for (size_t sample = 0; sample < order.size(); ++sample) {
    TensorBatch batch = data.batch(order, sample, 1, config.backend);
    Tensor logits = config.model->forward(batch.first, false, config.seed);
    std::vector<float> values = logits.data();
    std::vector<float> targets = batch.second.data();
    float loss = data.loss(logits, batch.second).data().at(0);
    require(std::isfinite(loss), "Non-finite inference loss");
    total_loss += loss;

    if (data.regression) {
      require(values.size() == targets.size(),
              "Regression output size mismatch");
      Json::Array predicted, truth;
      for (size_t i = 0; i < values.size(); ++i) {
        require(std::isfinite(values[i]), "Non-finite regression output");
        predicted.emplace_back(double(values[i]));
        truth.emplace_back(double(targets[i]));
      }
      predictions.emplace_back(Json::Object{{"sample", double(sample)},
                                            {"predicted", std::move(predicted)},
                                            {"target", std::move(truth)},
                                            {"loss", double(loss)}});
      continue;
    }

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

  Json::Object report{{"format", "elman.inference"},
                      {"version", 1},
                      {"model", config.model->name()},
                      {"backend", backend_name(config.backend)},
                      {"loss", total_loss / order.size()},
                      {"predictions", std::move(predictions)}};
  if (data.regression) {
    report["rmse"] = std::sqrt(total_loss / order.size());
  } else {
    report["accuracy"] = static_cast<double>(correct) / count;
  }
  Json result(std::move(report));
  std::cout << result.dump() << '\n';
  if (!options.output.empty()) {
    result.write(options.output);
  }
  return 0;
}

}  // namespace elman
