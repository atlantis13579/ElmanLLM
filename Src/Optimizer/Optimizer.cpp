#include "Optimizer/Optimizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>
#include "Core/Graph.h"
#include "Core/Device.h"
#include "Core/TensorDispatch.h"
#include <utility>
#include <vector>

#include "Optimizer/OptimizerInternal.h"

namespace elman {
namespace {

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

constexpr int reduction_chunk = 4096;

template<class T, class Square>
float float_gradient_squares(const std::vector<T>& gradients, Square square) {
  float total = 0.f;
  for (size_t start = 0; start < gradients.size(); start += reduction_chunk) {
    float partial = 0.f;
    const size_t end = std::min(start + reduction_chunk, gradients.size());
    for (size_t i = start; i < end; ++i) partial += square(gradients[i]);
    total += partial;
  }
  return total;
}

// One thread handles each bounded row. The final reduction has at most one
// element per row, and validation copies only its scalar result back.
template<class ReduceRows>
Tensor reduce_device_rows(const Tensor& value, const ReduceRows& reduce, DType dtype) {
  const int count = static_cast<int>(value.numel());
  const int rows = count / reduction_chunk;
  const int tail = count % reduction_chunk;
  Tensor partial;
  if (rows) partial = reduce(contiguous_view(value, {rows, reduction_chunk}), 1);
  if (tail) {
    Tensor right = reduce(contiguous_view(value, {1, tail}, size_t(rows) * reduction_chunk), 1);
    partial = partial.defined() ? concat({partial, right}, 0) : right;
  }
  return partial.defined() ? reduce(partial, 0) : Tensor::zeros({}, value.backend(), false, dtype);
}

struct ResidentChecks {
  Tensor partials;
  std::vector<size_t> offsets;
  std::vector<std::string> names;
  void prepare(const std::vector<Parameter>& parameters) {
    offsets.push_back(0);
    for (const Parameter& parameter : parameters)
      offsets.push_back(offsets.back() + std::max(size_t(1), (parameter.value.numel() + 4095) / 4096));
    require(offsets.back() <= size_t(std::numeric_limits<int>::max() / 2), "Optimizer validation workspace is too large");
    DeviceScope owner(parameters.front().value.backend(), parameters.front().value.device());
    partials = Tensor::zeros({int(offsets.back()), 2}, parameters.front().value.backend());
    names.reserve(parameters.size());
  }
  void add(const Tensor& value, const Tensor& gradient, const Tensor& first, const Tensor& second, const std::string& name, bool norm) {
    const int chunks = std::max(1, int((value.numel() + 4095) / 4096));
    const size_t index = names.size();
    require(index + 1 < offsets.size() && offsets[index + 1] - offsets[index] == size_t(chunks), "Optimizer validation partition mismatch");
    // Disjoint views share one compact download buffer, avoiding a reduction
    // and concatenation launch for every parameter on each validation pass.
    Tensor output = contiguous_view(partials, {chunks, 2}, offsets[index] * 2);
    dispatch_tensor_into("optimizer_stats", output, {value, gradient, first, second}, {int(norm)});
    names.push_back(name);
  }
  float finish() {
    if (names.empty()) return 0;
    require(names.size() + 1 == offsets.size(), "Incomplete optimizer validation pass");
    DeviceScope owner(partials.backend(), partials.device());
    const std::vector<float> results = partials.data();
    float squares = 0;
    for (size_t i = 0; i < names.size(); ++i) {
      float parameter_squares = 0;
      for (size_t row = offsets[i]; row < offsets[i + 1]; ++row) {
        require(results[row * 2 + 1] == 0.f, "Invalid device optimizer value: " + names[i]);
        parameter_squares += results[row * 2];
      }
      squares += parameter_squares;
    }
    names.clear();
    return squares;
  }
};

void validate_resident_float32(const Tensor& value, bool nonnegative, const std::string& name) {
  if (!value.numel()) return;
  Tensor (*const row_max)(const Tensor&, int) = [](const Tensor& x, int axis) { return amax(x, axis); };
  Tensor maximum = reduce_device_rows(abs(value), row_max, value.dtype());
  require(std::isfinite(maximum.data().front()), "Non-finite device optimizer value: " + name);
  if (nonnegative) {
    Tensor (*const row_min)(const Tensor&, int) = [](const Tensor& x, int axis) { return amin(x, axis); };
    Tensor minimum = reduce_device_rows(value, row_min, value.dtype());
    require(minimum.data().front() >= 0.f, "Negative device optimizer second moment: " + name);
  }
}

}  // namespace

Optimizer::Optimizer(const Json& settings, const Model& model, bool default_device_optimizer) {
  name_ = settings.string_or("optimizer", "adam");
  settings_.learning_rate = settings.number_or("learning_rate", 0.001);
  settings_.beta1 = settings.number_or("beta1", 0.9);
  settings_.beta2 = settings.number_or("beta2", 0.999);
  settings_.epsilon = settings.number_or("epsilon", 1e-8);
  settings_.momentum = settings.number_or("momentum", 0);
  settings_.weight_decay = settings.number_or("weight_decay", 0);
  decay_matrix_parameters_only_ = settings.bool_or("decay_matrix_parameters_only", false);
  settings_.clip_norm = settings.number_or("clip_grad_norm", 1);
  require(name_ == "adam" || name_ == "adamw" || name_ == "sgd",
          "optimizer must be adam, adamw or sgd");
  require(settings_.learning_rate > 0 && settings_.epsilon > 0 &&
              settings_.beta1 >= 0 && settings_.beta1 < 1 &&
              settings_.beta2 >= 0 && settings_.beta2 < 1 &&
              settings_.momentum >= 0 && settings_.momentum < 1 &&
              settings_.weight_decay >= 0 && settings_.clip_norm >= 0,
          "Invalid optimizer settings");

  require(std::isfinite(settings_.learning_rate) && std::isfinite(settings_.epsilon) &&
              std::isfinite(settings_.weight_decay) && std::isfinite(settings_.clip_norm), "Invalid optimizer settings");
  state_.optimizer = name_;
  state_.resident_float32 = settings.bool_or("device_optimizer", default_device_optimizer);
  for (const Parameter& parameter : model.parameters()) {
    require(is_arithmetic_float(parameter.value.dtype()) || is_complex(parameter.value.dtype()) || is_low_precision_float(parameter.value.dtype()), "Optimizer requires floating or complex parameters");
    const bool wide = parameter.value.dtype() == DType::Float64 || is_complex(parameter.value.dtype());
    const bool low = is_low_precision_float(parameter.value.dtype());
    const bool resident = state_.uses_resident_float32(parameter.value);
    DeviceScope owner(parameter.value.backend(), parameter.value.device());
    state_.first_moment.emplace_back(wide || low || resident ? 0 : parameter.value.numel(), 0.0f);
    state_.second_moment.emplace_back(wide || low || resident ? 0 : parameter.value.numel(), 0.0f);
    state_.first_moment_double.push_back(wide ? Tensor::zeros(parameter.value.shape(), parameter.value.backend(), false, parameter.value.dtype()) : Tensor{});
    state_.second_moment_double.push_back(wide ? Tensor::zeros(parameter.value.shape(), parameter.value.backend(), false, parameter.value.dtype()) : Tensor{});
    state_.first_moment_float.push_back(low || resident ? Tensor::zeros(parameter.value.shape(),parameter.value.backend()) : Tensor{});
    state_.second_moment_float.push_back(low || resident ? Tensor::zeros(parameter.value.shape(),parameter.value.backend()) : Tensor{});
    state_.sgd_buffer_initialized.push_back(0);
  }
}

double Optimizer::update(Model& model, double rate) {
  const size_t count = model.parameters().size();
  require(state_.first_moment.size() == count && state_.second_moment.size() == count &&
              (state_.first_moment_double.empty() || state_.first_moment_double.size() == count) &&
              (state_.second_moment_double.empty() || state_.second_moment_double.size() == count) &&
              (state_.first_moment_float.empty() || state_.first_moment_float.size() == count) &&
              (state_.second_moment_float.empty() || state_.second_moment_float.size() == count) &&
              (state_.sgd_buffer_initialized.empty() || state_.sgd_buffer_initialized.size() == count),
          "Optimizer state does not match model parameters");
  require(state_.optimizer == name_, "Optimizer state algorithm mismatch");
  require(std::isfinite(rate) && rate >= 0., "Invalid optimizer step learning rate");
  require(state_.step < uint64_t(std::numeric_limits<int64_t>::max()), "Optimizer step out of range");
  graph::RecordingGuard recording;
  std::vector<std::vector<float>> gradients(count), proposed_narrow(count);
  std::vector<Tensor> tensor_gradients(count), proposed_tensors(count);
  const bool batch_validation = count && std::all_of(model.parameters().begin(), model.parameters().end(), [&](const Parameter& p) {
    return state_.uses_resident_float32(p.value) && p.value.backend() == Backend::CUDA &&
           p.value.device() == model.parameters().front().value.device();
  });
  ResidentChecks resident_checks;
  if (batch_validation) resident_checks.prepare(model.parameters());
  // The FP4/FP8 frontend exposes a binary64 reference norm, including E8M0 gradients.
  const bool wide_norm = std::any_of(model.parameters().begin(), model.parameters().end(), [](const Parameter& p) {
    const DType dtype = p.value.dtype();
    return dtype == DType::Float64 || dtype == DType::Complex128 || is_low_precision_float(dtype);
  });
  float norm_squared = 0.f;
  double wide_norm_squared = 0.;
  for (size_t i = 0; i < count; ++i) {
    const Parameter& parameter = model.parameters()[i];
    const Tensor& value = parameter.value;
    const bool low = is_low_precision_float(value.dtype());
    const bool resident = state_.uses_resident_float32(value);
    float parameter_squares = 0.f;
    require((is_arithmetic_float(value.dtype()) || is_complex(value.dtype()) || low) && value.node()->gradient &&
                Tensor(value.node()->gradient).shape() == value.shape() &&
                Tensor(value.node()->gradient).dtype() == value.dtype() &&
                Tensor(value.node()->gradient).backend() == value.backend(),
            "Missing or invalid parameter gradient: " + parameter.name);
    require(low || resident || is_complex(value.dtype()) || ((state_.first_moment_float.empty() || !state_.first_moment_float[i].defined()) &&
                (state_.second_moment_float.empty() || !state_.second_moment_float[i].defined()) &&
                (state_.sgd_buffer_initialized.empty() || state_.sgd_buffer_initialized[i] == 0)), "Unexpected low-format optimizer state");
    if (resident) {
      DeviceScope owner(value.backend(), value.device());
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
              state_.first_moment_float.size() == count && state_.second_moment_float.size() == count &&
              (state_.first_moment_double.empty() || !state_.first_moment_double[i].defined()) &&
              (state_.second_moment_double.empty() || !state_.second_moment_double[i].defined()) &&
              (state_.sgd_buffer_initialized.empty() || state_.sgd_buffer_initialized[i] == 0),
              "Invalid device optimizer state: " + parameter.name);
      for (const Tensor* moment : {&state_.first_moment_float[i], &state_.second_moment_float[i]}) {
        require(moment->defined() && moment->dtype() == DType::Float32 && moment->shape() == value.shape() &&
                moment->backend() == value.backend() && moment->device() == value.device() && !moment->requires_grad(),
                "Invalid device optimizer moment: " + parameter.name);
        if (!batch_validation) validate_resident_float32(*moment, moment == &state_.second_moment_float[i], parameter.name);
      }
      if (!batch_validation) validate_resident_float32(value, false, parameter.name);
      tensor_gradients[i] = Tensor(value.node()->gradient).detach();
      if (batch_validation) resident_checks.add(value, tensor_gradients[i], state_.first_moment_float[i], state_.second_moment_float[i], parameter.name, true);
      else if (value.numel()) {
        const Tensor& gradient = tensor_gradients[i];
        Tensor (*const row_sum)(const Tensor&, int) = [](const Tensor& x, int axis) { return sum(x, axis); };
        Tensor squares = reduce_device_rows(mul(gradient, gradient), row_sum, DType::Float32);
        parameter_squares = squares.data().front();
      }
    } else if (is_complex(value.dtype())) {
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
          state_.first_moment_double.size() == count && state_.second_moment_double.size() == count &&
          (state_.first_moment_float.empty() || !state_.first_moment_float[i].defined()) &&
          (state_.second_moment_float.empty() || !state_.second_moment_float[i].defined()) &&
          state_.sgd_buffer_initialized.size() == count && state_.sgd_buffer_initialized[i] <= 1 &&
          (name_ == "sgd" || state_.sgd_buffer_initialized[i] == 0), "Invalid complex optimizer state storage");
      for (const Tensor* moment : {&state_.first_moment_double[i], &state_.second_moment_double[i]}) {
        require(moment->defined() && moment->dtype() == value.dtype() && moment->shape() == value.shape() &&
            moment->backend() == value.backend() && !moment->requires_grad(), "Invalid complex optimizer moment descriptor");
        for (const std::complex<double> z : moment->complex_data()) require(std::isfinite(z.real()) && std::isfinite(z.imag()) &&
            (moment != &state_.second_moment_double[i] || (z.real() >= 0. && z.imag() >= 0.)), "Invalid complex optimizer moment value");
      }
      for (const std::complex<double> z : value.complex_data()) require(std::isfinite(z.real()) && std::isfinite(z.imag()), "Non-finite complex parameter");
      tensor_gradients[i] = Tensor(value.node()->gradient).detach();
      const std::vector<std::complex<double>> components = tensor_gradients[i].complex_data();
      if (value.dtype() == DType::Complex128) {
        for (const std::complex<double>& z : components) wide_norm_squared += std::norm(z);
      } else {
        parameter_squares = float_gradient_squares(components, [](const std::complex<double>& z) {
          return std::norm(std::complex<float>(z));
        });
      }
    } else if (value.dtype() == DType::Float64) {
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
                  state_.first_moment_double.size() == count && state_.second_moment_double.size() == count,
              "Double optimizer state storage mismatch: " + parameter.name);
      for (const Tensor* moment : {&state_.first_moment_double[i], &state_.second_moment_double[i]}) {
        require(moment->defined() && moment->dtype() == DType::Float64 && moment->shape() == value.shape() &&
                    moment->backend() == value.backend() && !moment->requires_grad(), "Invalid Double optimizer moment: " + parameter.name);
        for (double x : moment->double_data()) require(std::isfinite(x) &&
            (moment != &state_.second_moment_double[i] || x >= 0.), "Invalid Double optimizer moment value: " + parameter.name);
      }
      for (double x : value.double_data()) require(std::isfinite(x), "Non-finite parameter: " + parameter.name);
      tensor_gradients[i] = Tensor(value.node()->gradient).detach();
      for (double x : tensor_gradients[i].double_data()) {
        require(std::isfinite(x), "Non-finite parameter gradient: " + parameter.name);
        wide_norm_squared += x * x;
      }
    } else if (low) {
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
          (state_.first_moment_double.empty() || !state_.first_moment_double[i].defined()) &&
          (state_.second_moment_double.empty() || !state_.second_moment_double[i].defined()) &&
          state_.first_moment_float.size()==count && state_.second_moment_float.size()==count &&
          state_.sgd_buffer_initialized.size()==count && state_.sgd_buffer_initialized[i]<=1 &&
          (name_=="sgd" || state_.sgd_buffer_initialized[i]==0), "Invalid low-format optimizer state storage: " + parameter.name);
      for(const Tensor* moment:{&state_.first_moment_float[i],&state_.second_moment_float[i]}) {
        require(moment->defined() && moment->dtype()==DType::Float32 && moment->shape()==value.shape() &&
            moment->backend()==value.backend() && !moment->requires_grad(), "Invalid low-format optimizer moment: " + parameter.name);
        for(float x:moment->data())require(std::isfinite(x) && (moment!=&state_.second_moment_float[i] || x>=0.f), "Invalid low-format optimizer moment value: " + parameter.name);
      }
      for(float x:value.data())require(std::isfinite(x), "Non-finite low-format parameter: " + parameter.name);
      tensor_gradients[i]=Tensor(value.node()->gradient).detach();
      for (float x : tensor_gradients[i].data()) wide_norm_squared += static_cast<double>(x) * x;
    } else {
      require((state_.first_moment_double.empty() || !state_.first_moment_double[i].defined()) &&
                  (state_.second_moment_double.empty() || !state_.second_moment_double[i].defined()), "Unexpected Double optimizer state");
      gradients[i] = value.grad();
      require(gradients[i].size() == value.numel() && state_.first_moment[i].size() == value.numel() &&
                  state_.second_moment[i].size() == value.numel(), "Optimizer state shape mismatch: " + parameter.name);
      parameter_squares = float_gradient_squares(gradients[i], [](float x) { return x * x; });
    }
    if (wide_norm) wide_norm_squared += parameter_squares;
    else norm_squared += parameter_squares;
  }
  norm_squared += resident_checks.finish();
  const double norm = wide_norm ? std::sqrt(wide_norm_squared) : std::sqrt(norm_squared);
  require(std::isfinite(norm), "Non-finite optimizer gradient norm");
  double scale = 1.;
  if (settings_.clip_norm > 0.) {
    scale = wide_norm ? std::min(1., settings_.clip_norm / (norm + 1e-6))
                      : std::min(1.f, float(settings_.clip_norm) / (float(norm) + 1e-6f));
  }
  TrainingState next = state_;
  ++next.step;
  const optimizer_internal::StepParameters step{rate, scale, settings_.weight_decay, settings_.momentum,
      settings_.beta1, settings_.beta2, settings_.epsilon,
      1 - std::pow(settings_.beta1, static_cast<double>(next.step)),
      1 - std::pow(settings_.beta2, static_cast<double>(next.step))};
  const FloatOptimizerParameters float_step{step.rate, step.gradient_scale, step.weight_decay, step.momentum,
      step.beta1, step.beta2, step.epsilon, step.correction1, std::sqrt(step.correction2)};
  // Prepare every update before mutating parameters or advancing any state.
  for (size_t i = 0; i < count; ++i) {
    const Parameter& parameter = model.parameters()[i];
    auto parameter_step = step;
    auto parameter_float_step = float_step;
    if (decay_matrix_parameters_only_ && parameter.value.shape().size() < 2) {
      parameter_step.weight_decay = 0;
      parameter_float_step.decay = 0;
      parameter_float_step.decay_factor = 1;
    }
    if (state_.uses_resident_float32(parameter.value)) {
      DeviceScope owner(parameter.value.backend(), parameter.value.device());
      optimizer_internal::TensorStep result = optimizer_internal::resident_float32_step(name_, parameter.value.detach(), tensor_gradients[i],
          state_.first_moment_float[i], state_.second_moment_float[i], parameter_step);
      if (batch_validation) resident_checks.add(result.value, result.value, result.first_moment, result.second_moment, parameter.name, false);
      else {
        validate_resident_float32(result.value, false, parameter.name);
        validate_resident_float32(result.first_moment, false, parameter.name);
        if (name_ != "sgd") validate_resident_float32(result.second_moment, true, parameter.name);
      }
      proposed_tensors[i] = std::move(result.value);
      next.first_moment_float[i] = std::move(result.first_moment);
      next.second_moment_float[i] = std::move(result.second_moment);
    } else if (is_complex(parameter.value.dtype())) {
      optimizer_internal::TensorStep result = optimizer_internal::complex_step(name_, parameter.value.detach(), tensor_gradients[i],
          state_.first_moment_double[i], state_.second_moment_double[i], parameter_step, !state_.sgd_buffer_initialized[i]);
      for (const Tensor* tensor : {&result.value, &result.first_moment, &result.second_moment})
        for (const std::complex<double> z : tensor->complex_data()) require(std::isfinite(z.real()) && std::isfinite(z.imag()), "Non-finite complex optimizer update");
      proposed_tensors[i] = std::move(result.value);
      next.first_moment_double[i] = std::move(result.first_moment);
      next.second_moment_double[i] = std::move(result.second_moment);
      if (name_ == "sgd") next.sgd_buffer_initialized[i] = 1;
    } else if (parameter.value.dtype() == DType::Float64) {
      optimizer_internal::TensorStep result = optimizer_internal::double_step(name_, parameter.value.detach(), tensor_gradients[i],
          state_.first_moment_double[i], state_.second_moment_double[i], parameter_step);
      for (const Tensor* tensor : {&result.value, &result.first_moment, &result.second_moment})
        for (double x : tensor->double_data()) require(std::isfinite(x), "Non-finite Double optimizer update: " + parameter.name);
      proposed_tensors[i] = std::move(result.value);
      next.first_moment_double[i] = std::move(result.first_moment);
      next.second_moment_double[i] = std::move(result.second_moment);
    } else if (is_low_precision_float(parameter.value.dtype())) {
      optimizer_internal::LowStep result=optimizer_internal::low_step(name_,parameter.value.detach(),tensor_gradients[i],
          state_.first_moment_float[i],state_.second_moment_float[i],parameter_step,!state_.sgd_buffer_initialized[i]);
      for(const Tensor* tensor:{&result.value_float32,&result.value,&result.first_moment,&result.second_moment})
        for(float x:tensor->data())require(std::isfinite(x), "Non-finite low-format optimizer update: " + parameter.name);
      proposed_tensors[i]=std::move(result.value);
      next.first_moment_float[i]=std::move(result.first_moment);
      next.second_moment_float[i]=std::move(result.second_moment);
      if(name_=="sgd")next.sgd_buffer_initialized[i]=1;
    } else {
      std::vector<float>& values = proposed_narrow[i];
      values = parameter.value.data();
      if (name_ == "sgd") optimizer_internal::sgd_step(values, gradients[i], next.first_moment[i], parameter_float_step);
      else if (name_ == "adam") optimizer_internal::adam_step(values, gradients[i], next.first_moment[i], next.second_moment[i], parameter_float_step);
      else optimizer_internal::adamw_step(values, gradients[i], next.first_moment[i], next.second_moment[i], parameter_float_step);
      for (size_t j = 0; j < values.size(); ++j) {
        if (!std::isfinite(values[j]) || !std::isfinite(next.first_moment[i][j]) ||
            !std::isfinite(next.second_moment[i][j]))
          throw std::runtime_error("Non-finite optimizer update: " + parameter.name);
      }
    }
  }
  resident_checks.finish();
  for (size_t i = 0; i < count; ++i) {
    Tensor& value = model.parameters()[i].value;
    if (proposed_tensors[i].defined()) {
      DeviceScope owner(value.backend(), value.device());
      dispatch_tensor_into("copy", value, {proposed_tensors[i]});
      ++value.node()->value->version;
    } else value.assign(proposed_narrow[i]);
  }
  state_ = std::move(next);
  return norm;
}

}  // namespace elman
