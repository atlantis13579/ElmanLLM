#include "Training/trainer.h"

#include "IO/checkpoint.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace elman {
namespace {

using Clock = std::chrono::steady_clock;
void require(bool condition, const std::string& message) { if (!condition) throw std::runtime_error(message); }
std::string resolve(const std::filesystem::path& base, const std::string& path) {
    auto result = std::filesystem::path(path);
    if (result.is_relative()) result = base / result;
    return std::filesystem::absolute(result).lexically_normal().string();
}
struct Configuration {
    Json json, definition;
    std::filesystem::path directory;
    Backend backend;
    uint32_t seed;
    std::unique_ptr<Model> model;
    explicit Configuration(const std::string& path, const RunOptions& options) {
        directory = std::filesystem::absolute(path).parent_path();
        json = Json::read(path);
        require(json.at("format").string() == "elman.training" && json.at("version").integer() == 1, "Unsupported training configuration format/version");
        definition = Json::read(resolve(directory, json.at("model").string()));
        backend = parse_backend(options.backend.empty() ? json.string_or("backend", "cpu") : options.backend);
        require(backend_available(backend), "Requested backend is unavailable: " + backend_name(backend));
        int seed_value = json.at("training").int_or("seed", 42);
        require(seed_value >= 0, "Seed must be non-negative");
        seed = static_cast<uint32_t>(seed_value);
        model = create_model(definition, backend, seed);
        require(json.at("model_name").string() == model->name(), "Training model_name does not match model definition");
    }
    std::string output(const std::string& key, const std::string& fallback) const {
        return resolve(directory, json.contains("output") ? json.at("output").string_or(key, fallback) : fallback);
    }
};

struct Dataset {
    std::vector<std::vector<float>> inputs, targets;
    Shape sample_shape;
    bool language = false;
    int classes = 0;
    explicit Dataset(const Configuration& config) {
        const auto& settings = config.json.at("data");
        sample_shape = config.model->sample_shape();
        language = config.model->architecture() == "gpt2";
        if (language && settings.contains("sequence_length")) {
            int length = settings.at("sequence_length").integer();
            require(length > 0 && length <= sample_shape[0], "Dataset sequence_length exceeds the model context");
            sample_shape[0] = length;
        }
        classes = config.model->output_classes();
        require(settings.at("format").string() == (language ? "token_csv" : "image_csv"), "Dataset format does not match model architecture");
        std::string path = resolve(config.directory, settings.at("path").string());
        std::ifstream in(path);
        require(static_cast<bool>(in), "Cannot open dataset: " + path);
        int limit = settings.int_or("limit_samples", 0);
        require(limit >= 0, "limit_samples must be non-negative");
        size_t sample_size = 1;
        for (int dim : sample_shape) sample_size *= static_cast<size_t>(dim);
        std::string line;
        size_t line_number = 0;
        while (std::getline(in, line)) {
            ++line_number;
            auto first = line.find_first_not_of(" \t\r");
            if (first == std::string::npos || line[first] == '#') continue;
            std::vector<float> row;
            size_t begin = 0;
            while (begin <= line.size()) {
                size_t comma = line.find(',', begin);
                std::string field = line.substr(begin, comma == std::string::npos ? comma : comma - begin);
                size_t consumed = 0;
                float value;
                try { value = std::stof(field, &consumed); }
                catch (const std::exception&) { throw std::runtime_error("Invalid dataset number at line " + std::to_string(line_number)); }
                require(field.find_first_not_of(" \t\r", consumed) == std::string::npos && std::isfinite(value), "Invalid dataset number at line " + std::to_string(line_number));
                row.push_back(value);
                if (comma == std::string::npos) break;
                begin = comma + 1;
            }
            require(row.size() == sample_size + 1, "Dataset line " + std::to_string(line_number) + " requires " + std::to_string(sample_size + 1) + " values");
            if (language) {
                for (float token : row) require(token >= 0 && token < classes && std::floor(token) == token, "Token ID outside vocabulary at dataset line " + std::to_string(line_number));
                inputs.emplace_back(row.begin(), row.end() - 1);
                targets.emplace_back(row.begin() + 1, row.end());
            } else {
                require(row[0] >= 0 && row[0] < classes && std::floor(row[0]) == row[0], "Invalid image class at dataset line " + std::to_string(line_number));
                inputs.emplace_back(row.begin() + 1, row.end());
                targets.push_back({row[0]});
            }
            if (limit && inputs.size() >= static_cast<size_t>(limit)) break;
        }
        require(!inputs.empty(), "Dataset contains no samples");
        require(in.eof() || static_cast<bool>(in), "Dataset read failed");
    }
    std::pair<Tensor, Tensor> batch(const std::vector<size_t>& order, size_t offset, size_t count, Backend backend) const {
        std::vector<float> input_values, target_values;
        input_values.reserve(count * inputs[0].size());
        target_values.reserve(count * targets[0].size());
        for (size_t i = 0; i < count; ++i) {
            size_t index = order[(offset + i) % order.size()];
            input_values.insert(input_values.end(), inputs[index].begin(), inputs[index].end());
            target_values.insert(target_values.end(), targets[index].begin(), targets[index].end());
        }
        Shape shape = sample_shape;
        shape.insert(shape.begin(), static_cast<int>(count));
        return {Tensor::from_data(shape, input_values, backend, false, language ? DType::Int32 : DType::Float32),
                Tensor::from_data({static_cast<int>(target_values.size())}, target_values, backend, false, DType::Int32)};
    }
};

Tensor classification_loss(const Tensor& logits, const Tensor& targets, int classes) {
    return cross_entropy(reshape(logits, {static_cast<int>(logits.numel() / classes), classes}), targets);
}
float evaluate(Configuration& config, const Dataset& data, const std::vector<size_t>& order, int batch_size) {
    double total = 0;
    for (size_t offset = 0; offset < order.size(); offset += batch_size) {
        size_t count = std::min(static_cast<size_t>(batch_size), order.size() - offset);
        auto batch = data.batch(order, offset, count, config.backend);
        auto output = config.model->forward(batch.first, false, config.seed);
        float loss = classification_loss(output, batch.second, data.classes).data().at(0);
        require(std::isfinite(loss), "Non-finite evaluation loss");
        total += count * loss;
    }
    return static_cast<float>(total / order.size());
}

uint32_t next_random(uint32_t& state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5; return state;
}
void shuffle(std::vector<size_t>& indices, uint32_t& state) {
    for (size_t i = indices.size(); i > 1; --i) std::swap(indices[i - 1], indices[next_random(state) % i]);
}

struct Optimizer {
    std::string name;
    double learning_rate, beta1, beta2, epsilon, momentum, weight_decay, clip_norm;
    TrainingState state;
    Optimizer(const Json& settings, Model& model) {
        name = settings.string_or("optimizer", "adam");
        learning_rate = settings.number_or("learning_rate", 0.001);
        beta1 = settings.number_or("beta1", 0.9);
        beta2 = settings.number_or("beta2", 0.999);
        epsilon = settings.number_or("epsilon", 1e-8);
        momentum = settings.number_or("momentum", 0);
        weight_decay = settings.number_or("weight_decay", 0);
        clip_norm = settings.number_or("clip_grad_norm", 1);
        require(name == "adam" || name == "adamw" || name == "sgd", "optimizer must be adam, adamw or sgd");
        require(learning_rate > 0 && epsilon > 0 && beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1 && momentum >= 0 && momentum < 1 && weight_decay >= 0 && clip_norm >= 0, "Invalid optimizer settings");
        state.optimizer = name;
        for (const auto& parameter : model.parameters()) {
            state.first_moment.emplace_back(parameter.value.numel(), 0);
            state.second_moment.emplace_back(parameter.value.numel(), 0);
        }
    }
    double update(Model& model, double rate) {
        std::vector<std::vector<float>> gradients;
        double norm_squared = 0;
        for (auto& parameter : model.parameters()) {
            auto gradient = parameter.value.grad();
            require(gradient.size() == parameter.value.numel(), "Missing parameter gradient: " + parameter.name);
            for (float g : gradient) {
                require(std::isfinite(g), "Non-finite parameter gradient: " + parameter.name);
                norm_squared += static_cast<double>(g) * g;
            }
            gradients.push_back(std::move(gradient));
        }
        double norm = std::sqrt(norm_squared);
        double scale = clip_norm > 0 && norm > clip_norm ? clip_norm / norm : 1;
        ++state.step;
        double correction1 = 1 - std::pow(beta1, static_cast<double>(state.step));
        double correction2 = 1 - std::pow(beta2, static_cast<double>(state.step));
        for (size_t i = 0; i < model.parameters().size(); ++i) {
            auto& parameter = model.parameters()[i];
            auto values = parameter.value.data();
            for (size_t j = 0; j < values.size(); ++j) {
                double gradient = gradients[i][j] * scale;
                if (name != "adamw") gradient += weight_decay * values[j];
                double update;
                if (name == "sgd") {
                    state.first_moment[i][j] = static_cast<float>(momentum * state.first_moment[i][j] + gradient);
                    update = state.first_moment[i][j];
                } else {
                    state.first_moment[i][j] = static_cast<float>(beta1 * state.first_moment[i][j] + (1 - beta1) * gradient);
                    state.second_moment[i][j] = static_cast<float>(beta2 * state.second_moment[i][j] + (1 - beta2) * gradient * gradient);
                    update = (state.first_moment[i][j] / correction1) / (std::sqrt(state.second_moment[i][j] / correction2) + epsilon);
                }
                if (name == "adamw") update += weight_decay * values[j];
                values[j] = static_cast<float>(values[j] - rate * update);
                require(std::isfinite(values[j]), "Non-finite optimizer update: " + parameter.name);
            }
            parameter.value.assign(values);
        }
        return norm;
    }
};

Json array(const std::vector<int>& numbers) {
    Json::Array values;
    for (int n : numbers) values.emplace_back(n);
    return Json(std::move(values));
}

}

int train(const std::string& configuration, const RunOptions& options) {
    Configuration config(configuration, options);
    Dataset data(config);
    const auto& settings = config.json.at("training");
    int steps = options.steps > 0 ? options.steps : settings.at("steps").integer();
    int batch_size = settings.int_or("batch_size", 1);
    int log_every = settings.int_or("log_every", 1);
    int checkpoint_every = settings.int_or("checkpoint_every", 0);
    int warmup = settings.int_or("warmup_steps", 0);
    std::string schedule = settings.string_or("lr_schedule", "constant");
    double min_ratio = settings.number_or("min_lr_ratio", 0.1);
    require(steps > 0 && batch_size > 0 && log_every > 0 && checkpoint_every >= 0 && warmup >= 0 && min_ratio >= 0 && min_ratio <= 1, "Invalid training step/batch/schedule settings");
    require(schedule == "constant" || schedule == "cosine", "lr_schedule must be constant or cosine");
    Optimizer optimizer(settings, *config.model);
    if (!options.resume.empty()) {
        load_checkpoint(options.resume, *config.model, config.definition, &optimizer.state);
        require(optimizer.state.optimizer == optimizer.name, "Resume optimizer differs from training configuration");
    }
    std::string checkpoint_path = options.checkpoint.empty() ? config.output("checkpoint", "../Intermediate/checkpoint.elman") : options.checkpoint;
    std::vector<size_t> order(data.inputs.size());
    std::iota(order.begin(), order.end(), 0);
    std::vector<std::vector<float>> initial_parameters;
    for (const auto& parameter : config.model->parameters()) initial_parameters.push_back(parameter.value.data());
    uint64_t dispatches_before = backend_dispatch_count(config.backend);
    float initial_loss = evaluate(config, data, order, batch_size);
    std::cout << "model=" << config.model->name() << " backend=" << backend_name(config.backend)
              << " parameters=" << config.model->parameter_count() << " samples=" << data.inputs.size()
              << " initial_eval_loss=" << std::setprecision(8) << initial_loss << std::endl;
    auto start = Clock::now();
    Json::Array history;
    uint32_t random_state = config.seed ? config.seed : 1;
    size_t offset = 0;
    bool shuffled = settings.bool_or("shuffle", true);
    for (int step = 0; step < steps; ++step) {
        for (auto& parameter : config.model->parameters()) parameter.value.zero_grad();
        std::vector<size_t> selected;
        selected.reserve(batch_size);
        for (int sample = 0; sample < batch_size; ++sample) {
            if (offset == 0 && shuffled) shuffle(order, random_state);
            selected.push_back(order[offset]);
            offset = (offset + 1) % order.size();
        }
        auto batch = data.batch(selected, 0, static_cast<size_t>(batch_size), config.backend);
        auto logits = config.model->forward(batch.first, true, config.seed + static_cast<uint32_t>(optimizer.state.step) * 31);
        auto loss = classification_loss(logits, batch.second, data.classes);
        float loss_value = loss.data().at(0);
        require(std::isfinite(loss_value), "Non-finite training loss at step " + std::to_string(step + 1));
        if (step == 0 && config.json.contains("output") && config.json.at("output").contains("graph")) {
            auto graph_path = config.output("graph", "../Intermediate/graph.json");
            Json::parse(loss.graph_json()).write(graph_path);
        }
        loss.backward();
        double rate = optimizer.learning_rate;
        if (warmup > 0 && step < warmup) rate *= static_cast<double>(step + 1) / warmup;
        else if (schedule == "cosine") {
            double progress = static_cast<double>(std::max(0, step - warmup)) / std::max(1, steps - warmup - 1);
            rate *= min_ratio + (1 - min_ratio) * 0.5 * (1 + std::cos(3.14159265358979323846 * progress));
        }
        double norm = optimizer.update(*config.model, rate);
        double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
        history.emplace_back(Json::Object{{"step", static_cast<double>(optimizer.state.step)}, {"loss", static_cast<double>(loss_value)}, {"gradient_norm", norm}, {"learning_rate", rate}, {"elapsed_seconds", elapsed}});
        if ((step + 1) % log_every == 0 || step + 1 == steps)
            std::cout << '[' << step + 1 << '/' << steps << "] step=" << optimizer.state.step << " loss=" << loss_value
                      << " grad_norm=" << norm << " lr=" << rate << " elapsed=" << std::fixed << std::setprecision(3) << elapsed << "s\n" << std::defaultfloat << std::setprecision(8) << std::flush;
        if (checkpoint_every > 0 && (step + 1) % checkpoint_every == 0) save_checkpoint(checkpoint_path, *config.model, config.definition, optimizer.state);
    }
    std::iota(order.begin(), order.end(), 0);
    float final_loss = evaluate(config, data, order, batch_size);
    size_t changed_parameters = 0;
    double maximum_change = 0;
    for (size_t i = 0; i < initial_parameters.size(); ++i) {
        auto final_values = config.model->parameters()[i].value.data();
        bool changed = false;
        for (size_t j = 0; j < final_values.size(); ++j) {
            double change = std::abs(static_cast<double>(final_values[j]) - initial_parameters[i][j]);
            maximum_change = std::max(maximum_change, change);
            changed = changed || change > 0;
        }
        if (changed) ++changed_parameters;
    }
    require(changed_parameters > 0, "Training failed to update any model parameter");
    std::cout << "saving_checkpoint=" << checkpoint_path << std::endl;
    save_checkpoint(checkpoint_path, *config.model, config.definition, optimizer.state);
    std::cout << "verifying_checkpoint_inference" << std::endl;
    auto roundtrip = create_model(config.definition, config.backend, config.seed + 1);
    load_checkpoint(checkpoint_path, *roundtrip, config.definition);
    auto verify_batch = data.batch(order, 0, 1, config.backend);
    auto expected = config.model->forward(verify_batch.first, false, config.seed).data();
    auto restored = roundtrip->forward(verify_batch.first, false, config.seed).data();
    double roundtrip_error = 0;
    require(expected.size() == restored.size(), "Checkpoint inference shape changed");
    for (size_t i = 0; i < expected.size(); ++i) {
        require(std::isfinite(expected[i]) && std::isfinite(restored[i]), "Non-finite checkpoint inference roundtrip output");
        roundtrip_error = std::max(roundtrip_error, std::abs(static_cast<double>(expected[i]) - restored[i]));
    }
    require(roundtrip_error <= 1e-6, "Checkpoint inference roundtrip mismatch");
    uint64_t dispatches = backend_dispatch_count(config.backend) - dispatches_before;
    require(dispatches > 0, "Training did not dispatch backend kernels");
    Json metrics(Json::Object{{"format", "elman.metrics"}, {"version", 1}, {"model", config.model->name()}, {"backend", backend_name(config.backend)},
                            {"initial_eval_loss", static_cast<double>(initial_loss)}, {"final_eval_loss", static_cast<double>(final_loss)},
                            {"steps", static_cast<double>(optimizer.state.step)}, {"parameters", static_cast<double>(config.model->parameter_count())},
                            {"samples", static_cast<double>(data.inputs.size())}, {"batch_size", batch_size},
                            {"input_shape", array(data.sample_shape)},
                            {"changed_parameter_tensors", static_cast<double>(changed_parameters)}, {"parameter_tensors", static_cast<double>(initial_parameters.size())},
                            {"max_parameter_change", maximum_change}, {"backend_dispatches", static_cast<double>(dispatches)},
                            {"checkpoint_roundtrip_max_error", roundtrip_error}, {"checkpoint", checkpoint_path}, {"history", std::move(history)}});
    metrics.write(options.output.empty() ? config.output("metrics", "../Intermediate/metrics.json") : options.output);
    std::cout << "final_eval_loss=" << final_loss << " changed_parameters=" << changed_parameters << '/' << initial_parameters.size()
              << " backend_dispatches=" << dispatches << " checkpoint_roundtrip_error=" << roundtrip_error << " checkpoint=" << checkpoint_path << '\n';
    require(!settings.bool_or("require_loss_decrease", false) || final_loss < initial_loss, "Smoke validation requires final evaluation loss below initial loss");
    return 0;
}

int infer(const std::string& configuration, const RunOptions& options) {
    Configuration config(configuration, options);
    Dataset data(config);
    std::string checkpoint = options.checkpoint.empty() ? config.output("checkpoint", "../Intermediate/checkpoint.elman") : options.checkpoint;
    load_checkpoint(checkpoint, *config.model, config.definition);
    std::vector<size_t> order(data.inputs.size());
    std::iota(order.begin(), order.end(), 0);
    Json::Array predictions;
    size_t correct = 0, count = 0;
    double total_loss = 0;
    for (size_t i = 0; i < order.size(); ++i) {
        auto batch = data.batch(order, i, 1, config.backend);
        auto logits = config.model->forward(batch.first, false, config.seed);
        auto values = logits.data();
        auto targets = batch.second.data();
        float loss = classification_loss(logits, batch.second, data.classes).data().at(0);
        require(std::isfinite(loss), "Non-finite inference loss");
        total_loss += loss;
        std::vector<int> predicted, truth;
        for (size_t row = 0; row < targets.size(); ++row) {
            int best = 0;
            for (int cls = 0; cls < data.classes; ++cls) {
                require(std::isfinite(values[row * data.classes + cls]), "Non-finite inference output");
                if (values[row * data.classes + cls] > values[row * data.classes + best]) best = cls;
            }
            predicted.push_back(best); truth.push_back(static_cast<int>(targets[row]));
            if (best == truth.back()) ++correct;
            ++count;
        }
        predictions.emplace_back(Json::Object{{"sample", static_cast<double>(i)}, {"predicted", array(predicted)}, {"target", array(truth)}, {"loss", static_cast<double>(loss)}});
    }
    Json result(Json::Object{{"format", "elman.inference"}, {"version", 1}, {"model", config.model->name()}, {"backend", backend_name(config.backend)},
                           {"loss", total_loss / order.size()}, {"accuracy", static_cast<double>(correct) / count}, {"predictions", std::move(predictions)}});
    std::cout << result.dump() << '\n';
    if (!options.output.empty()) result.write(options.output);
    return 0;
}

int validate_json(const std::string& path) {
    auto document = Json::read(path);
    std::cout << document.dump() << '\n';
    return 0;
}

}
