#include "Training/Trainer.h"
#include "Fusion/Fusion.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <utility>

#include "Core/Autocast.h"
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Core/Device.h"
#include "Core/Graph.h"
#include "Distributed/ProcessGroup.h"
#include "IO/Checkpoint.h"
#include "Optimizer/Optimizer.h"
#include "Training/DataLoader.h"
#include "Training/LossScaler.h"
#include "Training/Sampler.h"
#ifdef ELMAN_HAS_CUDA
#include "Backends/CudaGemm.h"
#endif

namespace elman {
namespace {

using Clock = std::chrono::steady_clock;
void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}
float evaluate(Model& model, const BatchDataset& data, const DataLoader& loader, uint32_t seed,
               const std::vector<size_t>& order, int batch_size, ProcessGroup& group) {
  double total = 0;
  const size_t begin = order.size() * size_t(group.rank()) / group.size();
  const size_t end = order.size() * size_t(group.rank() + 1) / group.size();
  for (size_t offset = begin; offset < end; offset += batch_size) {
    size_t count =
        std::min(static_cast<size_t>(batch_size), end - offset);
    TensorBatch batch = loader.batch(order, offset, count);
    Tensor output = model.forward(batch.first, false, seed);
    float loss =
        data.loss(output, batch.second).data().at(0);
    require(std::isfinite(loss), "Non-finite evaluation loss");
    total += count * loss;
  }
  group.all_reduce_sum(&total, 1);
  return static_cast<float>(total / order.size());
}

void synchronize_parameters(Model& model, ProcessGroup& group) {
  if (group.size() == 1) return;
  LocalFunction broadcast{[&](Tensor& value) {
    std::vector<uint8_t> bytes = group.rank() == 0 ? value.raw_data() : std::vector<uint8_t>(value.nbytes());
    group.broadcast(bytes.data(), bytes.size());
    if (group.rank() != 0) value.assign_raw(bytes);
  }};
  for (Parameter& parameter : model.parameters()) broadcast(parameter.value);
  for (Parameter& buffer : model.checkpoint_buffers()) broadcast(buffer.value);
}

void synchronize_gradients(Model& model, ProcessGroup& group) {
  if (group.size() == 1) return;
  for (Parameter& parameter : model.parameters()) {
    Tensor& value = parameter.value;
    require(value.dtype() == DType::Float32 && value.node()->gradient,
            "Data parallel training requires a Float32 gradient for " + parameter.name);
    Tensor gradient = Tensor(value.node()->gradient).detach();
    std::vector<float> local = gradient.data();
    group.all_reduce_sum(local.data(), local.size());
    gradient.assign(local);
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

int train_model(Model& model, const BatchDataset& data, const TrainingRun& run, ProcessGroup& group,
                const BatchDataset* validation) {
  const bool primary = group.rank() == 0;
  const Json& settings = run.settings;
  AllocationPool pool(run.backend, size_t(settings.int_or("pool_megabytes", 3072)) * 1024 * 1024,
                      settings.bool_or("reuse_memory", false));
  ExecutionStream stream(run.backend);
  StreamScope stream_scope(stream, settings.bool_or("asynchronous", false));
  const std::string& checkpoint_path = run.checkpoint;
  require(!checkpoint_path.empty() && !run.metrics.empty(), "Training requires checkpoint and metrics destinations");
  DataLoader loader(data, run.backend);
  const int planned_steps = settings.at("steps").integer();
  const int steps = run.steps > 0 ? run.steps : planned_steps;
  int batch_size = settings.int_or("batch_size", 1);
  int accumulation = settings.int_or("gradient_accumulation_steps", 1);
  int log_every = settings.int_or("log_every", 1);
  int checkpoint_every = settings.int_or("checkpoint_every", 0);
  const bool profile_steps = settings.bool_or("profile_steps", false);
  const int eval_every = settings.int_or("eval_every", 0);
  const int eval_batches = settings.int_or("eval_batches", 0);
  require(eval_every >= 0 && eval_batches >= 0, "Invalid evaluation settings");
  int warmup = settings.int_or("warmup_steps", 0);
  std::string schedule = settings.string_or("lr_schedule", "constant");
  double min_ratio = settings.number_or("min_lr_ratio", 0.1);
  bool shuffled = settings.bool_or("shuffle", true);
  const DType precision = parse_dtype(settings.string_or("precision", "float32"));
  fusion::Options fusion_options;
  const bool gpu_fusion = run.backend == Backend::CUDA || run.backend == Backend::Metal;
  const bool legacy_fusion = run.backend != Backend::CUDA || settings.bool_or("cuda_fusion", true);
  fusion_options.enabled = gpu_fusion && settings.bool_or("operator_fusion", legacy_fusion);
  fusion_options.verify = settings.bool_or("operator_fusion_verify", settings.bool_or("cuda_fusion_verify", true));
  fusion_options.cache_directory = settings.string_or("operator_fusion_cache", settings.string_or("cuda_fusion_cache", "Intermediate/Fusion"));
  fusion::Policy fusion_policy(fusion_options);
  const auto fusion_before = fusion::statistics();
  require(is_compute_float(precision), "Training precision must be float32, float16 or bfloat16");
  const std::string matmul_precision = settings.string_or("matmul_precision", "fp32");
  const std::string convolution_precision = settings.string_or("convolution_precision", "fp32");
  require(matmul_precision == "fp32" || matmul_precision == "tf32",
          "matmul_precision must be fp32 or tf32");
  require(convolution_precision == "fp32" || convolution_precision == "tf32",
          "convolution_precision must be fp32 or tf32");
  require((matmul_precision != "tf32" && convolution_precision != "tf32") ||
              run.backend == Backend::CUDA,
          "TF32 math requires the CUDA backend");
#ifdef ELMAN_HAS_CUDA
  if (run.backend == Backend::CUDA)
    set_cuda_tf32_math(matmul_precision == "tf32", convolution_precision == "tf32");
#endif
  const bool scaled = precision == DType::Float16;
  const float initial_scale = static_cast<float>(settings.number_or("initial_loss_scale", 65536));
  const int growth_interval = settings.int_or("loss_scale_growth_interval", 2000);
  LossScaler scaler(scaled, initial_scale, growth_interval);
  require(steps > 0 && planned_steps > 0 && batch_size >= group.size() && accumulation > 0 && log_every > 0 &&
              checkpoint_every >= 0 && warmup >= 0 && min_ratio >= 0 &&
              min_ratio <= 1,
          "Invalid training step/batch/schedule settings");
  require(schedule == "constant" || schedule == "cosine",
          "lr_schedule must be constant or cosine");
  Optimizer optimizer(settings, model, run.backend == Backend::CUDA);
  BatchSampler sampler(data.size(), run.seed, shuffled);
  Json::Object resume_settings{
      {"dataset", data.fingerprint()}, {"input_shape", array(data.shape())},
      {"samples", std::to_string(data.size())},
      {"seed", double(run.seed)}, {"batch_size", batch_size},
      {"accumulation", accumulation}, {"shuffle", shuffled}, {"world_size", group.size()},
      {"precision", dtype_name(precision)}, {"initial_loss_scale", double(initial_scale)},
      {"loss_scale_growth_interval", growth_interval},
      {"device_optimizer", optimizer.state().resident_float32},
      {"schedule", schedule}, {"warmup", warmup},
      {"schedule_steps", schedule == "cosine" ? planned_steps : 0},
      {"min_lr_ratio", min_ratio}, {"learning_rate", optimizer.learning_rate()},
      {"momentum", settings.number_or("momentum", 0)},
      {"weight_decay", settings.number_or("weight_decay", 0)},
      {"beta1", settings.number_or("beta1", 0.9)}, {"beta2", settings.number_or("beta2", 0.999)},
      {"epsilon", settings.number_or("epsilon", 1e-8)}, {"clip_grad_norm", settings.number_or("clip_grad_norm", 1)}};
  if (settings.bool_or("decay_matrix_parameters_only", false))
    resume_settings["decay_matrix_parameters_only"] = true;
  if (matmul_precision == "tf32") resume_settings["matmul_precision"] = matmul_precision;
  if (convolution_precision == "tf32") resume_settings["convolution_precision"] = convolution_precision;
  const std::string resume_configuration = Json(std::move(resume_settings)).dump(0);
  if (group.size() > 1) {
    const std::string contract = resume_configuration + '\n' + optimizer.name() + '\n' +
                                 run.definition.dump(0) + '\n' + std::to_string(steps) + '\n' +
                                 backend_name(run.backend) + '\n' + (run.resume.empty() ? "fresh" : "resume");
    std::string primary_contract = contract;
    group.broadcast(primary_contract.data(), primary_contract.size());
    require(primary_contract == contract, "Training configuration differs across ranks");
  }
  uint64_t checkpoint_checksum = 0;
  uint64_t attempted_steps = 0;
  if (!run.resume.empty()) {
    TrainingProgress restored;
    load_checkpoint(run.resume, model, run.definition,
                    &optimizer.state(), &restored, data.size());
    checkpoint_checksum = restored.checksum;
    attempted_steps = restored.attempted_steps;
    require(optimizer.state().optimizer == optimizer.name(),
            "Resume optimizer differs from training configuration");
    if (restored.configuration.empty() && primary) {
      std::cerr << "Legacy checkpoint restored without dataset progress\n";
    } else if (!restored.configuration.empty()) {
      require(restored.configuration == resume_configuration,
              "Resume dataset or training settings differ from checkpoint");
      sampler.restore(std::move(restored.sampler));
      scaler.restore(restored.loss_scale);
    }
  }
  if (group.size() > 1) {
    uint64_t primary_checksum = checkpoint_checksum;
    group.broadcast(&primary_checksum, sizeof(primary_checksum));
    require(primary_checksum == checkpoint_checksum, "Training checkpoint differs across ranks");
  }
  synchronize_parameters(model, group);
  require(attempted_steps <= uint64_t(std::numeric_limits<int64_t>::max()) - uint64_t(steps),
          "Training step exceeds the checkpoint range");
  LocalFunction save_to{[&](const std::string& destination) {
    TrainingProgress progress{sampler.state(), resume_configuration};
    progress.attempted_steps = attempted_steps;
    progress.loss_scale = scaler.state();
    save_checkpoint(destination, model, run.definition, optimizer.state(), &progress);
  }};
  std::vector<size_t> order(data.size());
  std::iota(order.begin(), order.end(), 0);
  if (eval_batches && order.size() > size_t(eval_batches) * batch_size) {
    std::mt19937 rng(run.seed);
    std::shuffle(order.begin(), order.end(), rng);
    order.resize(size_t(eval_batches) * batch_size);
  }
  std::vector<size_t> validation_order(validation ? validation->size() : 0);
  std::iota(validation_order.begin(), validation_order.end(), 0);
  if (eval_batches && validation_order.size() > size_t(eval_batches) * batch_size) {
    std::mt19937 rng(run.seed + 1);
    std::shuffle(validation_order.begin(), validation_order.end(), rng);
    validation_order.resize(size_t(eval_batches) * batch_size);
  }
  LocalFunction validation_loss{[&] {
    DataLoader held_out(*validation, run.backend);
    return evaluate(model, *validation, held_out, run.seed, validation_order, batch_size, group);
  }};
  double best_validation = std::numeric_limits<double>::infinity();
  const double initial_validation = validation ? validation_loss() : 0;
  Json::Array validation_history;
  uint64_t dispatches_before = backend_dispatch_count(run.backend);
  float initial_loss = evaluate(model, data, loader, run.seed, order, batch_size, group);
  if (primary) std::cout << "model=" << model.name()
            << " backend=" << backend_name(run.backend)
            << " parameters=" << model.parameter_count()
            << " samples=" << data.size()
            << " initial_eval_loss=" << std::setprecision(8) << initial_loss
            << std::endl;
  if (validation && primary) std::cout << "initial_validation_loss=" << initial_validation << std::endl;
  std::chrono::steady_clock::time_point start = Clock::now();
  Json::Array history;
  const size_t local_begin = size_t(batch_size) * group.rank() / group.size();
  const size_t local_end = size_t(batch_size) * (group.rank() + 1) / group.size();
  const double local_weight = double(local_end - local_begin) / batch_size / accumulation;
  const bool defer_loss_read = run.backend == Backend::CUDA && settings.bool_or("asynchronous", false) && accumulation > 1;
  size_t skipped_steps = 0;
  for (int step = 0; step < steps; ++step) {
    const auto step_started = Clock::now();
    double load_seconds = 0, forward_seconds = 0, backward_seconds = 0;
    uint64_t load_dispatches = 0, forward_dispatches = 0, backward_dispatches = 0;
    for (Parameter& parameter : model.parameters()) {
      parameter.value.zero_grad();
    }
    const uint64_t global_step = optimizer.state().step;
    const float loss_scale = scaler.state().scale;
    double loss_value = 0;
    std::vector<Tensor> micro_losses;
    if (defer_loss_read) micro_losses.reserve(accumulation);
    for (int micro = 0; micro < accumulation; ++micro) {
      const uint64_t load_before = profile_steps ? backend_dispatch_count(run.backend) : 0;
      const auto load_started = Clock::now();
      TensorBatch batch = loader.next(sampler, static_cast<size_t>(batch_size), local_begin, local_end - local_begin);
      load_seconds += std::chrono::duration<double>(Clock::now() - load_started).count();
      if (profile_steps) load_dispatches += backend_dispatch_count(run.backend) - load_before;
      const uint64_t forward_before = profile_steps ? backend_dispatch_count(run.backend) : 0;
      const auto forward_started = Clock::now();
      const uint32_t seed = run.seed + static_cast<uint32_t>(attempted_steps) * 31u +
                            static_cast<uint32_t>(micro) * 0x9e3779b9u +
                            static_cast<uint32_t>(group.rank()) * 0x85ebca6bu;
      Tensor loss;
      {
        AutocastScope autocast(precision);
        Tensor logits = model.forward(batch.first, true, seed);
        if (precision != DType::Float32 && is_narrow_float(logits.dtype())) logits = cast(logits, DType::Float32);
        loss = data.loss(logits, batch.second);
      }
      float micro_loss = 0;
      if (defer_loss_read) micro_losses.push_back(reshape(loss.detach(), {1}));
      else micro_loss = loss.data().at(0);
      forward_seconds += std::chrono::duration<double>(Clock::now() - forward_started).count();
      if (profile_steps) forward_dispatches += backend_dispatch_count(run.backend) - forward_before;
      require(scaled || std::isfinite(micro_loss),
              "Non-finite training loss at step " + std::to_string(global_step + 1));
      loss_value += double(micro_loss) * local_weight;
      const uint64_t backward_before = profile_steps ? backend_dispatch_count(run.backend) : 0;
      const auto backward_started = Clock::now();
      if (primary && step == 0 && micro == 0 && !run.graph.empty()) {
        Json::parse(loss.graph_json()).write(run.graph);
      }
      if (local_weight == 1. && loss_scale == 1.f) {
        loss.backward();
      } else {
        loss.backward(Tensor::from_data(loss.shape(), {static_cast<float>(local_weight * loss_scale)}, run.backend));
      }
      backward_seconds += std::chrono::duration<double>(Clock::now() - backward_started).count();
      if (profile_steps) backward_dispatches += backend_dispatch_count(run.backend) - backward_before;
    }
    if (defer_loss_read) {
      // Read detached scalar losses together after all backward submissions.
      // Every loss is checked before the optimizer can modify parameters or state.
      for (float micro_loss : concat(micro_losses, 0).data()) {
        require(scaled || std::isfinite(micro_loss),
                "Non-finite training loss at step " + std::to_string(global_step + 1));
        loss_value += double(micro_loss) * local_weight;
      }
    }
    group.all_reduce_sum(&loss_value, 1);
    const bool gradients_finite = scaler.unscale_and_check(model);
    double overflow = gradients_finite && std::isfinite(loss_value) ? 0.0 : 1.0;
    if (scaled) group.all_reduce_sum(&overflow, 1);
    double rate = optimizer.learning_rate();
    if (global_step < uint64_t(warmup)) {
      rate *= static_cast<double>(global_step + 1) / warmup;
    } else if (schedule == "cosine") {
      const double progress = std::min(1., double(global_step - uint64_t(warmup)) /
                                               std::max(1, planned_steps - warmup - 1));
      rate *= min_ratio + (1 - min_ratio) * 0.5 *
                              (1 + std::cos(3.14159265358979323846 * progress));
    }
    double norm = 0;
    const uint64_t optimizer_before = profile_steps ? backend_dispatch_count(run.backend) : 0;
    const auto optimizer_started = Clock::now();
    if (overflow == 0) {
      synchronize_gradients(model, group);
      norm = optimizer.update(model, rate);
    } else {
      ++skipped_steps;
    }
    const double optimizer_seconds = std::chrono::duration<double>(Clock::now() - optimizer_started).count();
    const uint64_t optimizer_dispatches = profile_steps ? backend_dispatch_count(run.backend) - optimizer_before : 0;
    ++attempted_steps;
    scaler.update(overflow == 0);
    if (validation && eval_every && ((step + 1) % eval_every == 0 || step + 1 == steps)) {
      const double value = validation_loss();
      if (primary) {
        validation_history.emplace_back(Json::Object{{"step", double(optimizer.state().step)}, {"loss", value}});
        std::cout << "validation step=" << optimizer.state().step << " loss=" << value << std::endl;
        if (value < best_validation) {
          best_validation = value;
          if (!run.best_checkpoint.empty()) save_to(run.best_checkpoint);
        }
      }
    }
    if (!primary) continue;
    double elapsed =
        std::chrono::duration<double>(Clock::now() - start).count();
    Json::Object step_metrics{{"step", static_cast<double>(optimizer.state().step)},
                     {"loss", std::isfinite(loss_value) ? Json(loss_value) : Json()},
                     {"gradient_norm", norm},
                     {"loss_scale", double(loss_scale)}, {"skipped", overflow != 0},
                     {"learning_rate", rate},
                     {"elapsed_seconds", elapsed}};
    if (profile_steps) {
      step_metrics["data_seconds"] = load_seconds;
      step_metrics["data_dispatches"] = double(load_dispatches);
      step_metrics["forward_seconds"] = forward_seconds;
      step_metrics["forward_dispatches"] = double(forward_dispatches);
      step_metrics["backward_seconds"] = backward_seconds;
      step_metrics["backward_dispatches"] = double(backward_dispatches);
      step_metrics["optimizer_seconds"] = optimizer_seconds;
      step_metrics["optimizer_dispatches"] = double(optimizer_dispatches);
      step_metrics["total_seconds"] = std::chrono::duration<double>(Clock::now() - step_started).count();
    }
    history.emplace_back(std::move(step_metrics));
    if ((step + 1) % log_every == 0 || step + 1 == steps) {
      std::cout << '[' << step + 1 << '/' << steps
                << "] step=" << optimizer.state().step << " loss=" << loss_value
                << " grad_norm=" << norm << " lr=" << rate
                << " elapsed=" << std::fixed << std::setprecision(3) << elapsed
                << "s\n"
                << std::defaultfloat << std::setprecision(8) << std::flush;
    }
    if (checkpoint_every > 0 && (step + 1) % checkpoint_every == 0) {
      save_to(checkpoint_path);
    }
  }
  float final_loss = evaluate(model, data, loader, run.seed, order, batch_size, group);
  const double final_validation = validation ? validation_loss() : 0;
  if (!primary) return 0;
  if (validation && final_validation < best_validation) {
    best_validation = final_validation;
    if (!run.best_checkpoint.empty()) save_to(run.best_checkpoint);
  }
  std::cout << "saving_checkpoint=" << checkpoint_path << std::endl;
  save_to(checkpoint_path);
  uint64_t dispatches =
      backend_dispatch_count(run.backend) - dispatches_before;
  const auto fusion_after = fusion::statistics();
  Json metrics(Json::Object{
      {"format", "elman.metrics"},
      {"version", 1},
      {"model", model.name()},
      {"backend", backend_name(run.backend)},
      {"initial_eval_loss", static_cast<double>(initial_loss)},
      {"final_eval_loss", static_cast<double>(final_loss)},
      {"steps", static_cast<double>(optimizer.state().step)},
      {"attempted_steps", static_cast<double>(attempted_steps)}, {"skipped_steps", static_cast<double>(skipped_steps)},
      {"precision", dtype_name(precision)},
      {"matmul_precision", matmul_precision},
      {"convolution_precision", convolution_precision},
      {"operator_fusion", fusion::enabled()},
      {"cuda_fusion", run.backend == Backend::CUDA && fusion::enabled()},
      {"fusion_compiled", double(fusion_after.compiled-fusion_before.compiled)},
      {"fusion_cache_hits", double(fusion_after.cache_hits-fusion_before.cache_hits)},
      {"fusion_launches", double(fusion_after.launches-fusion_before.launches)},
      {"fusion_commands", double(fusion_after.fused_commands-fusion_before.fused_commands)},
      {"fusion_verified", double(fusion_after.verified-fusion_before.verified)},
      {"fusion_rejected", double(fusion_after.rejected-fusion_before.rejected)},
      {"fusion_compile_failures", double(fusion_after.compile_failures-fusion_before.compile_failures)},
      {"fusion_elided_stores", double(fusion_after.elided_stores-fusion_before.elided_stores)},
      {"nvidia_tf32_override", std::getenv("NVIDIA_TF32_OVERRIDE") ? Json(std::getenv("NVIDIA_TF32_OVERRIDE")) : Json()},
      {"parameters", static_cast<double>(model.parameter_count())},
      {"samples", static_cast<double>(data.size())},
      {"batch_size", batch_size},
      {"gradient_accumulation_steps", accumulation},
      {"world_size", group.size()},
      {"input_shape", array(data.shape())},
      {"parameter_tensors", static_cast<double>(model.parameters().size())},
      {"backend_dispatches", static_cast<double>(dispatches)},
      {"checkpoint", checkpoint_path},
      {"history", std::move(history)}});
  if (validation) {
    auto report = metrics.object();
    report["initial_validation_loss"] = initial_validation;
    report["final_validation_loss"] = final_validation;
    report["best_validation_loss"] = best_validation;
    report["validation_history"] = std::move(validation_history);
    report["validation_samples"] = double(validation_order.size());
    report["best_checkpoint"] = run.best_checkpoint;
    metrics = Json(std::move(report));
  }
  metrics.write(run.metrics);
  std::cout << "final_eval_loss=" << final_loss
            << " backend_dispatches=" << dispatches
            << " checkpoint=" << checkpoint_path << '\n';
  require(!settings.bool_or("require_loss_decrease", false) ||
              final_loss < initial_loss,
          "Final evaluation loss must be below the initial loss");
  return 0;
}

}  // namespace elman
