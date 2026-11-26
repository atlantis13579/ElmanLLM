#pragma once

#include "Core/Tensor.h"
#include "IO/Json.h"
#include <memory>

namespace elman {

struct Parameter {
    std::string name;
    Tensor value;
};

class Model {
public:
    virtual ~Model() = default;
    virtual Tensor forward(const Tensor& input, bool training, uint32_t seed = 1) = 0;
    virtual Shape sample_shape() const = 0;
    virtual int output_classes() const = 0;
    virtual bool language() const { return false; }
    virtual bool regression() const { return false; }
    virtual std::vector<Parameter> checkpoint_buffers() const { return {}; }
    const std::string& name() const { return name_; }
    const std::string& architecture() const { return architecture_; }
    std::vector<Parameter>& parameters() { return parameters_; }
    const std::vector<Parameter>& parameters() const { return parameters_; }
    size_t parameter_count() const;
protected:
    Model(std::string name, std::string architecture, Backend backend, uint32_t seed);
    Tensor parameter(const std::string& name, const Shape& shape, float standard_deviation);
    Tensor constant_parameter(const std::string& name, const Shape& shape, float value);
    std::string name_, architecture_;
    Backend backend_;
    uint32_t seed_;
    std::vector<Parameter> parameters_;
};

std::unique_ptr<Model> create_model(const Json& definition, Backend backend, uint32_t seed = 42);
// Resolve relative JSON includes into a self-contained checkpoint definition.
Json read_model_definition(const std::filesystem::path& path);

}
