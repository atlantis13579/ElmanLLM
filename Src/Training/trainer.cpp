#include "Training/Trainer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <utility>

#include "IO/Checkpoint.h"
#include "Optimizer/Optimizer.h"
#include "Runtime/RunContext.h"

namespace elman {
namespace {

using Clock = std::chrono::steady_clock;
void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
float evaluate(Configuration& config, const Dataset& data,
               const std::vector<size_t>& order, int batch_size) {
  double total = 0;
  for (size_t offset = 0; offset < order.size(); offset += batch_size) {
    size_t count =
        std::min(static_cast<size_t>(batch_size), order.size() - offset);
    auto batch = data.batch(order, offset, count, config.backend);
    auto output = config.model->forward(batch.first, false, config.seed);
    float loss =
        classification_loss(output, batch.second, data.classes).data().at(0);
    require(std::isfinite(loss), "Non-finite evaluation loss");
    total += count * loss;
  }
  return static_cast<float>(total / order.size());
}

uint32_t next_random(uint32_t& state) {
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return state;
}
void shuffle(std::vector<size_t>& indices, uint32_t& state) {
  for (size_t i = indices.size(); i > 1; --i) {
    std::swap(indices[i - 1], indices[next_random(state) % i]);
  }
}

Json array(const std::vector<int>& numbers) {
  Json::Array values;
  for (int n : numbers) {
    values.emplace_back(n);
  }
  return Json(std::move(values));
}

}  // namespace

int train(const std::string& configuration, const RunOptions& options) {
  Configuration config(configuration, options);
  Dataset data(config);
  const auto& settings = config.json.at("training");
  int steps =
      options.steps > 0 ? options.steps : settings.at("steps").integer();
  int batch_size = settings.int_or("batch_size", 1);
  int log_every = settings.int_or("log_every", 1);
  int checkpoint_every = settings.int_or("checkpoint_every", 0);
  int warmup = settings.int_or("warmup_steps", 0);
  std::string schedule = settings.string_or("lr_schedule", "constant");
  double min_ratio = settings.number_or("min_lr_ratio", 0.1);
  require(steps > 0 && batch_size > 0 && log_every > 0 &&
              checkpoint_every >= 0 && warmup >= 0 && min_ratio >= 0 &&
              min_ratio <= 1,
          "Invalid training step/batch/schedule settings");
  require(schedule == "constant" || schedule == "cosine",
          "lr_schedule must be constant or cosine");
  Optimizer optimizer(settings, *config.model);
  if (!options.resume.empty()) {
    load_checkpoint(options.resume, *config.model, config.definition,
                    &optimizer.state());
    require(optimizer.state().optimizer == optimizer.name(),
            "Resume optimizer differs from training configuration");
  }
  std::string checkpoint_path =
      options.checkpoint.empty()
          ? config.output("checkpoint", "../Intermediate/checkpoint.elman")
          : options.checkpoint;
  std::vector<size_t> order(data.inputs.size());
  std::iota(order.begin(), order.end(), 0);
  std::vector<std::vector<float>> initial_parameters;
  for (const auto& parameter : config.model->parameters()) {
    initial_parameters.push_back(parameter.value.data());
  }
  uint64_t dispatches_before = backend_dispatch_count(config.backend);
  float initial_loss = evaluate(config, data, order, batch_size);
  std::cout << "model=" << config.model->name()
            << " backend=" << backend_name(config.backend)
            << " parameters=" << config.model->parameter_count()
            << " samples=" << data.inputs.size()
            << " initial_eval_loss=" << std::setprecision(8) << initial_loss
            << std::endl;
  auto start = Clock::now();
  Json::Array history;
  uint32_t random_state = config.seed ? config.seed : 1;
  size_t offset = 0;
  bool shuffled = settings.bool_or("shuffle", true);
  for (int step = 0; step < steps; ++step) {
    for (auto& parameter : config.model->parameters()) {
      parameter.value.zero_grad();
    }
    std::vector<size_t> selected;
    selected.reserve(batch_size);
    for (int sample = 0; sample < batch_size; ++sample) {
      if (offset == 0 && shuffled) {
        shuffle(order, random_state);
      }
      selected.push_back(order[offset]);
      offset = (offset + 1) % order.size();
    }
    auto batch = data.batch(selected, 0, static_cast<size_t>(batch_size),
                            config.backend);
    auto logits = config.model->forward(
        batch.first, true,
        config.seed + static_cast<uint32_t>(optimizer.state().step) * 31);
    auto loss = classification_loss(logits, batch.second, data.classes);
    float loss_value = loss.data().at(0);
    require(std::isfinite(loss_value),
            "Non-finite training loss at step " + std::to_string(step + 1));
    if (step == 0 && config.json.contains("output") &&
        config.json.at("output").contains("graph")) {
      auto graph_path = config.output("graph", "../Intermediate/graph.json");
      Json::parse(loss.graph_json()).write(graph_path);
    }
    loss.backward();
    double rate = optimizer.learning_rate();
    if (warmup > 0 && step < warmup) {
      rate *= static_cast<double>(step + 1) / warmup;
    } else if (schedule == "cosine") {
      double progress = static_cast<double>(std::max(0, step - warmup)) /
                        std::max(1, steps - warmup - 1);
      rate *= min_ratio + (1 - min_ratio) * 0.5 *
                              (1 + std::cos(3.14159265358979323846 * progress));
    }
    double norm = optimizer.update(*config.model, rate);
    double elapsed =
        std::chrono::duration<double>(Clock::now() - start).count();
    history.emplace_back(
        Json::Object{{"step", static_cast<double>(optimizer.state().step)},
                     {"loss", static_cast<double>(loss_value)},
                     {"gradient_norm", norm},
                     {"learning_rate", rate},
                     {"elapsed_seconds", elapsed}});
    if ((step + 1) % log_every == 0 || step + 1 == steps) {
      std::cout << '[' << step + 1 << '/' << steps
                << "] step=" << optimizer.state().step << " loss=" << loss_value
                << " grad_norm=" << norm << " lr=" << rate
                << " elapsed=" << std::fixed << std::setprecision(3) << elapsed
                << "s\n"
                << std::defaultfloat << std::setprecision(8) << std::flush;
    }
    if (checkpoint_every > 0 && (step + 1) % checkpoint_every == 0) {
      save_checkpoint(checkpoint_path, *config.model, config.definition,
                      optimizer.state());
    }
  }
  std::iota(order.begin(), order.end(), 0);
  float final_loss = evaluate(config, data, order, batch_size);
  size_t changed_parameters = 0;
  double maximum_change = 0;
  for (size_t i = 0; i < initial_parameters.size(); ++i) {
    auto final_values = config.model->parameters()[i].value.data();
    bool changed = false;
    for (size_t j = 0; j < final_values.size(); ++j) {
      double change = std::abs(static_cast<double>(final_values[j]) -
                               initial_parameters[i][j]);
      maximum_change = std::max(maximum_change, change);
      changed = changed || change > 0;
    }
    if (changed) {
      ++changed_parameters;
    }
  }
  require(changed_parameters > 0,
          "Training failed to update any model parameter");
  std::cout << "saving_checkpoint=" << checkpoint_path << std::endl;
  save_checkpoint(checkpoint_path, *config.model, config.definition,
                  optimizer.state());
  std::cout << "verifying_checkpoint_inference" << std::endl;
  auto roundtrip =
      create_model(config.definition, config.backend, config.seed + 1);
  load_checkpoint(checkpoint_path, *roundtrip, config.definition);
  auto verify_batch = data.batch(order, 0, 1, config.backend);
  auto expected =
      config.model->forward(verify_batch.first, false, config.seed).data();
  auto restored =
      roundtrip->forward(verify_batch.first, false, config.seed).data();
  double roundtrip_error = 0;
  require(expected.size() == restored.size(),
          "Checkpoint inference shape changed");
  for (size_t i = 0; i < expected.size(); ++i) {
    require(std::isfinite(expected[i]) && std::isfinite(restored[i]),
            "Non-finite checkpoint inference roundtrip output");
    roundtrip_error =
        std::max(roundtrip_error,
                 std::abs(static_cast<double>(expected[i]) - restored[i]));
  }
  require(roundtrip_error <= 1e-6, "Checkpoint inference roundtrip mismatch");
  uint64_t dispatches =
      backend_dispatch_count(config.backend) - dispatches_before;
  require(dispatches > 0, "Training did not dispatch backend kernels");
  Json metrics(Json::Object{
      {"format", "elman.metrics"},
      {"version", 1},
      {"model", config.model->name()},
      {"backend", backend_name(config.backend)},
      {"initial_eval_loss", static_cast<double>(initial_loss)},
      {"final_eval_loss", static_cast<double>(final_loss)},
      {"steps", static_cast<double>(optimizer.state().step)},
      {"parameters", static_cast<double>(config.model->parameter_count())},
      {"samples", static_cast<double>(data.inputs.size())},
      {"batch_size", batch_size},
      {"input_shape", array(data.sample_shape)},
      {"changed_parameter_tensors", static_cast<double>(changed_parameters)},
      {"parameter_tensors", static_cast<double>(initial_parameters.size())},
      {"max_parameter_change", maximum_change},
      {"backend_dispatches", static_cast<double>(dispatches)},
      {"checkpoint_roundtrip_max_error", roundtrip_error},
      {"checkpoint", checkpoint_path},
      {"history", std::move(history)}});
  metrics.write(options.output.empty()
                    ? config.output("metrics", "../Intermediate/metrics.json")
                    : options.output);
  std::cout << "final_eval_loss=" << final_loss
            << " changed_parameters=" << changed_parameters << '/'
            << initial_parameters.size() << " backend_dispatches=" << dispatches
            << " checkpoint_roundtrip_error=" << roundtrip_error
            << " checkpoint=" << checkpoint_path << '\n';
  require(!settings.bool_or("require_loss_decrease", false) ||
              final_loss < initial_loss,
          "Smoke validation requires final evaluation loss below initial loss");
  return 0;
}

int validate_json(const std::string& path) {
  auto document = Json::read(path);
  std::cout << document.dump() << '\n';
  return 0;
}

}  // namespace elman
