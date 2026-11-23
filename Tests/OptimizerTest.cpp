#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

#include "Optimizer/Optimizer.h"

namespace {

class ScalarModel final : public elman::Model {
 public:
  explicit ScalarModel(elman::Backend backend, const std::vector<float>& weights = {2.f})
      : Model("Scalar", "scalar", backend, 1) {
    parameters_.push_back(
        {"weight", elman::Tensor::from_data({int(weights.size())}, weights, backend, true)});
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
  elman::Tensor input = elman::Tensor::from_data({1}, {3.0f}, backend);
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
  elman::Json configuration = settings("sgd", 1);
  elman::Json::Object object = configuration.object();
  object["weight_decay"] = 0;
  object["momentum"] = 0;
  elman::Optimizer optimizer(elman::Json(object), model);
  make_gradient(model, backend);
  near(optimizer.update(model, 0.1), 3.0, "Unclipped gradient norm");
  near(model.parameters()[0].value.data()[0], 1.9,
       "Clipped SGD parameter update");
}

void test_float32_trajectory(elman::Backend backend, bool resident) {
  // Three steps of torch.optim in float32 with foreach=False.
  struct Reference {
    std::string algorithm;
    std::vector<float> weight, first, second;
  };
  const Reference references[] = {
      {"sgd", {.936331511f, -1.83803523f}, {.0651204288f, .0962660015f}, {0.f, 0.f}},
      {"adam", {.861071706f, -1.83302784f},
       {resident ? .0143199228f : .0143199265f, resident ? -.0154254744f : -.0154254660f},
       {resident ? .0714016557f : .0714016631f, .127258360f}},
      {"adamw", {.835363865f, resident ? -1.74518025f : -1.74518037f},
       {-.0691086948f, .160605088f}, {resident ? .0707956702f : .0707956776f, .130752206f}}};
  const std::vector<float> gradients[] = {{.7f, -1.1f}, {-1.3f, 1.7f}, {.23f, .19f}};
  for (const Reference& reference : references) {
    ScalarModel model(backend, {1.f, -2.f});
    elman::Optimizer optimizer(elman::Json::Object{
        {"optimizer", reference.algorithm}, {"clip_grad_norm", 0}, {"weight_decay", .2}, {"momentum", .6},
        {"beta1", .81}, {"beta2", .967}, {"epsilon", 1e-5}}, model, resident);
    elman::Tensor& weight = model.parameters()[0].value;
    for (const std::vector<float>& gradient : gradients) {
      weight.zero_grad();
      weight.backward(elman::Tensor::from_data({2}, gradient, backend));
      optimizer.update(model, .17);
    }
    const elman::TrainingState& state = optimizer.state();
    const std::vector<float> first = resident ? state.first_moment_float[0].data() : state.first_moment[0];
    const std::vector<float> second = resident ? state.second_moment_float[0].data() : state.second_moment[0];
    if (weight.data() != reference.weight || first != reference.first || second != reference.second)
      throw std::runtime_error(reference.algorithm + " float32 trajectory differs from PyTorch");
    weight.zero_grad();
    weight.backward(elman::Tensor::ones({2}, backend));
    if (optimizer.update(model, 0.) != std::sqrt(2.f))
      throw std::runtime_error("Float32 gradient norm did not round to float32");
  }
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
    test_float32_trajectory(backend, false);
    if (backend == elman::Backend::CUDA) test_float32_trajectory(backend, true);
    std::cout << "Optimizer tests passed on " << elman::backend_name(backend)
              << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Optimizer test failure: " << error.what() << '\n';
    return 1;
  }
}
