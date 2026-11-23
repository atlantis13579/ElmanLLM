#include "Core/Tensor.h"
#include "Core/Graph.h"
#include "Core/Float16.h"
#include "Core/TensorDispatch.h"
#include <cstring>
#include <stdexcept>

namespace elman {
namespace {
void require(bool valid, const char* message) {
    if (!valid) throw std::invalid_argument(message);
}
std::vector<uint8_t> encode_complex(const std::vector<std::complex<double>>& values, DType dtype) {
    require(is_complex(dtype), "Complex host values require complex storage");
    const size_t component_bytes = dtype_storage_bytes(complex_component_dtype(dtype), 1);
    std::vector<uint8_t> result(dtype_storage_bytes(dtype, values.size()));
    for (size_t i = 0; i < values.size(); ++i) {
        const double components[2] = {values[i].real(), values[i].imag()};
        for (int part = 0; part < 2; ++part) {
            unsigned char* output = result.data() + (i * 2 + part) * component_bytes;
            if (dtype == DType::Complex128) std::memcpy(output, &components[part], sizeof(double));
            else if (dtype == DType::Complex64) {
                const float component = float(components[part]);
                std::memcpy(output, &component, sizeof(float));
            } else {
                const uint16_t component = numeric::float_to_half(float(components[part]));
                std::memcpy(output, &component, sizeof(uint16_t));
            }
        }
    }
    return result;
}
}
Tensor Tensor::from_complex_data(Shape shape, const std::vector<std::complex<double>>& values,
                                  Backend backend, bool gradient, DType dtype) {
    require(shape_element_count(shape) == values.size(), "Complex data and tensor shape sizes differ");
    return from_raw(std::move(shape), dtype, encode_complex(values, dtype), backend, gradient);
}
std::vector<std::complex<double>> Tensor::complex_data() const {
    require(is_complex(dtype()), "Complex data requires complex storage");
    const std::vector<uint8_t> bytes = raw_data();
    const size_t component_bytes = dtype_storage_bytes(complex_component_dtype(dtype()), 1);
    std::vector<std::complex<double>> values(numel());
    for (size_t i = 0; i < values.size(); ++i) {
        double components[2];
        for (int part = 0; part < 2; ++part) {
            const unsigned char* input = bytes.data() + (i * 2 + part) * component_bytes;
            if (dtype() == DType::Complex128) std::memcpy(&components[part], input, sizeof(double));
            else if (dtype() == DType::Complex64) {
                float component;
                std::memcpy(&component, input, sizeof(float));
                components[part] = component;
            } else {
                uint16_t component;
                std::memcpy(&component, input, sizeof(uint16_t));
                components[part] = numeric::half_to_float(component);
            }
        }
        values[i] = {components[0], components[1]};
    }
    return values;
}
void Tensor::assign_complex(const std::vector<std::complex<double>>& values) {
    require(numel() == values.size(), "Complex assignment shape mismatch");
    assign_raw(encode_complex(values, dtype()));
}
std::vector<std::complex<double>> Tensor::complex_grad() const {
    require(is_complex(dtype()), "Complex gradients require complex storage");
    return node()->gradient ? Tensor(node()->gradient).complex_data() : std::vector<std::complex<double>>{};
}
} // namespace elman
