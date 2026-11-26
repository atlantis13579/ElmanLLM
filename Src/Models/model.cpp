#include "Models/Model.h"
#include "Models/GraphModel.h"
#include <stdexcept>

namespace elman {
std::unique_ptr<Model> create_spatial_policy_value(const Json&, Backend, uint32_t);
std::unique_ptr<Model> create_continuous_actor_critic(const Json&, Backend, uint32_t);

Model::Model(std::string name, std::string architecture, Backend backend, uint32_t seed)
    : name_(std::move(name)), architecture_(std::move(architecture)), backend_(backend), seed_(seed) {}
Tensor Model::parameter(const std::string& name, const Shape& shape, float standard_deviation) {
    Tensor value = Tensor::normal(shape, 0, standard_deviation, seed_++, backend_, true);
    parameters_.push_back({name, value});
    return value;
}
Tensor Model::constant_parameter(const std::string& name, const Shape& shape, float value) {
    size_t size = 1;
    for (int dim : shape) size *= static_cast<size_t>(dim);
    Tensor tensor = Tensor::from_data(shape, std::vector<float>(size, value), backend_, true);
    parameters_.push_back({name, tensor});
    return tensor;
}
size_t Model::parameter_count() const {
    size_t count = 0;
    for (const Parameter& parameter : parameters_) count += parameter.value.numel();
    return count;
}
std::unique_ptr<Model> create_model(const Json& definition, Backend backend, uint32_t seed) {
    if (definition.at("format").string() != "elman.model") throw std::invalid_argument("Unsupported model format");
    if (definition.at("version").integer() == 2) return std::make_unique<GraphModel>(definition, backend, seed);
    if (definition.at("version").integer() != 1) throw std::invalid_argument("Unsupported model version");
    const std::string architecture = definition.at("architecture").string();
    if (architecture == "spatial_policy_value") return create_spatial_policy_value(definition, backend, seed);
    if (architecture == "continuous_actor_critic") return create_continuous_actor_critic(definition, backend, seed);
    throw std::runtime_error("Unsupported model architecture: " + architecture);
}

}
