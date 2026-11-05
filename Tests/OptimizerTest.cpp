#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Optimizer/Optimizer.h"

namespace {

class ScalarModel final : public elman::Model {
 public:
  explicit ScalarModel(elman::Backend backend)
      : Model("Scalar", "scalar", backend, 1) {
    parameters_.push_back(
        {"weight", elman::Tensor::from_data({1}, {2.0f}, backend, true)});
  }

  elman::Tensor forward(const elman::Tensor& input, bool, uint32_t) override {
    return elman::mul(parameters_[0].value, input);
  }

  elman::Shape sample_shape() const override {
    return {1};
  }
  int output_classes() const override {
    return 1;
  }
};

void near(double actual, double expected, const std::string& description) {
  if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-6) {
    throw std::runtime_error(description + " differs from expected value");
  }
}

elman::Json settings(const std::string& algorithm, double clip_norm = 0) {
  return elman::Json::Object{
      {"optimizer", algorithm}, {"learning_rate", 0.1},
      {"beta1", 0.9},           {"beta2", 0.999},
      {"epsilon", 1e-8},        {"momentum", 0.5},
      {"weight_decay", 0.1},    {"clip_grad_norm", clip_norm}};
}

void make_gradient(ScalarModel& model, elman::Backend backend) {
  auto input = elman::Tensor::from_data({1}, {3.0f}, backend);
  elman::sum_all(model.forward(input, true, 1)).backward();
}

void test_algorithm(const std::string& algorithm, elman::Backend backend,
                    double expected_weight, double expected_first,
                    double expected_second) {
  ScalarModel model(backend);
  elman::Optimizer optimizer(settings(algorithm), model);
  make_gradient(model, backend);
  near(optimizer.update(model, optimizer.learning_rate()), 3.0,
       algorithm + " gradient norm");
  near(model.parameters()[0].value.data()[0], expected_weight,
       algorithm + " parameter update");
  near(optimizer.state().first_moment[0][0], expected_first,
       algorithm + " first moment");
  near(optimizer.state().second_moment[0][0], expected_second,
       algorithm + " second moment");
  if (optimizer.name() != algorithm || optimizer.state().step != 1 ||
      optimizer.state().optimizer != algorithm) {
    throw std::runtime_error(algorithm + " optimizer metadata mismatch");
  }

  if (algorithm == "sgd") {
    model.parameters()[0].value.zero_grad();
    make_gradient(model, backend);
    optimizer.update(model, optimizer.learning_rate());
    near(model.parameters()[0].value.data()[0], 1.2032,
         "SGD second momentum update");
    if (optimizer.state().step != 2) {
      throw std::runtime_error("SGD step count did not advance");
    }
  }
}

void test_clipping(elman::Backend backend) {
  ScalarModel model(backend);
  auto configuration = settings("sgd", 1);
  auto object = configuration.object();
  object["weight_decay"] = 0;
  object["momentum"] = 0;
  elman::Optimizer optimizer(elman::Json(object), model);
  make_gradient(model, backend);
  near(optimizer.update(model, 0.1), 3.0, "Unclipped gradient norm");
  near(model.parameters()[0].value.data()[0], 1.9,
       "Clipped SGD parameter update");
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) {
      throw std::runtime_error("Expected a CPU or CUDA backend argument");
    }
    elman::Backend backend = elman::parse_backend(argv[1]);
    if (!elman::backend_available(backend)) {
      return 77;
    }
    test_algorithm("sgd", backend, 1.68, 3.2, 0);
    test_algorithm("adam", backend, 1.9, 0.32, 0.01024);
    test_algorithm("adamw", backend, 1.88, 0.3, 0.009);
    test_clipping(backend);
    std::cout << "Optimizer tests passed on " << elman::backend_name(backend)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Optimizer test failure: " << error.what() << '\n';
    return 1;
  }
}
