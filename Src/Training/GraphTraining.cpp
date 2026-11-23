#include "Training/GraphTraining.h"
#include "Backends/CpuParallel.h"
#include "Core/Device.h"
#include "Inference/GraphSession.h"
#include "IO/AtomicFile.h"
#include "IO/NumPy.h"
#include "IO/Safetensors.h"
#include "Optimizer/Optimizer.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <set>
#include <stdexcept>

namespace elman {
namespace {
class GraphParameters final : public Model {
public:
    GraphParameters(Backend backend, uint32_t seed) : Model("Declarative graph", "tensor_graph", backend, seed) {}
    Tensor forward(const Tensor&, bool, uint32_t) override { throw std::logic_error("Use graph execution"); }
    Shape sample_shape() const override { return {}; }
    int output_classes() const override { return 0; }
};
Shape shape_of(const Json& value) {
    Shape shape;
    for (const auto& dim : value.array()) shape.push_back(dim.integer());
    return shape;
}
Tensor read_tensor(const Json& desc, const std::filesystem::path& root, Backend backend) {
    if (desc.contains("file")) {
        const std::string key = desc.string_or("key", "input");
        return load_tensor_archive(root / std::filesystem::u8path(desc.at("file").string()), backend, key).at(key);
    }
    const Shape shape = shape_of(desc.at("shape"));
    std::vector<float> values;
    for (const auto& value : desc.at("data").array()) values.push_back(float(value.number()));
    return Tensor::from_data(shape, values, backend);
}
}

int train_graph(const std::filesystem::path& configuration, const RunOptions& options) {
    const Json config = Json::read(configuration);
    if (config.string_or("format", "") != "elman.training_graph" || config.at("version").integer() != 1)
        throw std::invalid_argument("Unsupported training graph format or version");
    if (!options.resume.empty() || !options.checkpoint.empty() || !options.output.empty())
        throw std::invalid_argument("Training graph input/output paths are defined by its JSON");
    const auto root = std::filesystem::absolute(configuration).parent_path();
    const Json& training = config.at("training");
    const Backend backend = parse_backend(options.backend.empty() ? training.string_or("backend", "cpu") : options.backend);
    const int device = backend == Backend::CPU ? -1 : training.int_or("device", 0);
    DeviceScope owner(backend, device);
    const uint32_t seed = uint32_t(training.int_or("seed", 42));
    const int steps = options.steps ? options.steps : training.at("steps").integer();
    const int log_every = training.int_or("log_every", 10);
    const int threads = training.int_or("cpu_threads", 0);
    if (steps <= 0 || log_every <= 0 || threads < 0) throw std::invalid_argument("Invalid graph training schedule");
    set_cpu_thread_count(threads);
    GraphParameters model(backend, seed);
    NamedTensors weights;
    for (const auto& entry : config.at("tensors").object()) {
        Tensor value = read_tensor(entry.second, root, backend);
        if (entry.second.bool_or("trainable", false)) {
            if (value.dtype() != DType::Float32) throw std::invalid_argument("Graph parameters currently require Float32");
            value = Tensor::from_data(value.shape(), value.data(), backend, true);
            model.parameters().push_back({entry.first, value});
        }
        weights.emplace(entry.first, value);
    }
    if (model.parameters().empty()) throw std::invalid_argument("Training graph has no trainable parameters");
    const auto session = make_graph_session(config.at("graph"), weights, backend, device);
    Optimizer optimizer(training, model, true);
    std::mt19937 random(seed);
    auto inputs = [&]() {
        NamedTensors values;
        for (const auto& entry : config.at("inputs").object()) {
            const Json& desc = entry.second;
            if (desc.contains("random_indices")) {
                const Json& sample = desc.at("random_indices");
                const int count = sample.at("count").integer(), upper = sample.at("upper").integer();
                if (count <= 0 || upper <= 0) throw std::invalid_argument("Invalid random index source");
                std::uniform_int_distribution<int32_t> distribution(0, upper - 1);
                std::vector<int32_t> indices(size_t(count), 0);
                for (auto& index : indices) index = distribution(random);
                values.emplace(entry.first, Tensor::from_int_data({count}, indices, backend));
            } else values.emplace(entry.first, weights.at(desc.at("tensor").string()));
        }
        return values;
    };
    const std::string loss_name = training.string_or("loss", "loss");
    auto evaluate = [&]() {
        const auto saved = random;
        random.seed(seed + 1);
        double loss = 0;
        const int batches = training.int_or("evaluation_batches", 8);
        if (batches <= 0) throw std::invalid_argument("evaluation_batches must be positive");
        for (int i = 0; i < batches; ++i) loss += session.run(inputs()).at(loss_name).data().at(0);
        random = saved;
        return loss / batches;
    };
    const double initial = evaluate();
    Json::Array history;
    const auto start = std::chrono::steady_clock::now();
    for (int step = 0; step < steps; ++step) {
        for (auto& parameter : model.parameters()) parameter.value.zero_grad();
        auto result = session.run_with_grad(inputs());
        auto loss = result.at(loss_name);
        if (loss.numel() != 1 || !loss.requires_grad()) throw std::invalid_argument("Training loss must be a differentiable scalar");
        const float value = loss.data().at(0);
        if (!std::isfinite(value)) throw std::runtime_error("Non-finite graph loss");
        loss.backward();
        const double norm = optimizer.update(model, optimizer.learning_rate());
        if ((step + 1) % log_every == 0 || step == 0 || step + 1 == steps) {
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            history.emplace_back(Json::Object{{"step", step + 1}, {"loss", value}, {"gradient_norm", norm}, {"seconds", seconds}});
            std::cout << "[" << step + 1 << "/" << steps << "] loss=" << value << " seconds=" << seconds << std::endl;
        }
    }
    const double final = evaluate();
    if (training.bool_or("require_loss_decrease", false) && !(final < initial))
        throw std::runtime_error("Graph evaluation loss did not decrease; outputs were not replaced");
    const Json& output = config.at("output");
    const auto evaluated = session.run(inputs());
    for (const Json& file : output.at("tensors").array()) {
        const auto path = root / std::filesystem::u8path(file.at("file").string());
        NamedTensors selected;
        for (const auto& entry : file.at("values").object()) {
            const std::string source = entry.second.string();
            selected.emplace(entry.first, weights.count(source) ? weights.at(source) : evaluated.at(source));
        }
        if (path.extension() == ".npy" && selected.size() == 1) io::save_npy(path, selected.begin()->second);
        else if (path.extension() == ".safetensors") save_safetensors(path, selected);
        else throw std::invalid_argument("Graph outputs require single-tensor NPY or Safetensors");
    }
    const auto metrics = root / std::filesystem::u8path(output.at("metrics").string());
    Json report(Json::Object{{"format", "elman.graph_metrics"}, {"version", 1}, {"initial_loss", initial}, {"final_loss", final},
        {"steps", steps}, {"backend", backend_name(backend)}, {"history", std::move(history)}});
    write_file_atomically(metrics, [&](const std::filesystem::path& temporary) { report.write(temporary.u8string()); });
    std::cout << "initial_loss=" << initial << " final_loss=" << final << std::endl;
    return 0;
}
}
