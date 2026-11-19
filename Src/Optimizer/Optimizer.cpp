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
    auto right = reduce(contiguous_view(value, {1, tail}, size_t(rows) * reduction_chunk), 1);
    partial = partial.defined() ? concat({partial, right}, 0) : right;
  }
  return partial.defined() ? reduce(partial, 0) : Tensor::zeros({}, value.backend(), false, dtype);
}

void validate_resident_float32(const Tensor& value, bool nonnegative, const std::string& name) {
  if (!value.numel()) return;
  auto row_max = [](const Tensor& x, int axis) { return amax(x, axis); };
  require(std::isfinite(reduce_device_rows(abs(value), row_max, value.dtype()).data().front()),
          "Non-finite device optimizer value: " + name);
  if (nonnegative) {
    auto row_min = [](const Tensor& x, int axis) { return amin(x, axis); };
    require(reduce_device_rows(value, row_min, value.dtype()).data().front() >= 0.f,
            "Negative device optimizer second moment: " + name);
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
  for (const auto& parameter : model.parameters()) {
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
  const auto count = model.parameters().size();
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
  std::vector<Tensor> wide_gradients(count), proposed_wide(count);
  double norm_squared = 0.;
  for (size_t i = 0; i < count; ++i) {
    const auto& parameter = model.parameters()[i];
    const auto& value = parameter.value;
    const bool low = is_low_precision_float(value.dtype());
    const bool resident = state_.uses_resident_float32(value);
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
      for (const auto* moment : {&state_.first_moment_float[i], &state_.second_moment_float[i]}) {
        require(moment->defined() && moment->dtype() == DType::Float32 && moment->shape() == value.shape() &&
                moment->backend() == value.backend() && moment->device() == value.device() && !moment->requires_grad(),
                "Invalid device optimizer moment: " + parameter.name);
        validate_resident_float32(*moment, moment == &state_.second_moment_float[i], parameter.name);
      }
      validate_resident_float32(value, false, parameter.name);
      wide_gradients[i] = Tensor(value.node()->gradient).detach();
      if (value.numel()) {
        const auto wide = cast(wide_gradients[i], DType::Float64);
        auto row_sum = [](const Tensor& x, int axis) { return sum(x, axis); };
        const double squared = reduce_device_rows(mul(wide, wide), row_sum, DType::Float64).double_data().front();
        require(std::isfinite(squared), "Non-finite device optimizer gradient: " + parameter.name);
        norm_squared += squared;
      }
    } else if (is_complex(value.dtype())) {
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
          state_.first_moment_double.size() == count && state_.second_moment_double.size() == count &&
          (state_.first_moment_float.empty() || !state_.first_moment_float[i].defined()) &&
          (state_.second_moment_float.empty() || !state_.second_moment_float[i].defined()) &&
          state_.sgd_buffer_initialized.size() == count && state_.sgd_buffer_initialized[i] <= 1 &&
          (name_ == "sgd" || state_.sgd_buffer_initialized[i] == 0), "Invalid complex optimizer state storage");
      for (const auto* moment : {&state_.first_moment_double[i], &state_.second_moment_double[i]}) {
        require(moment->defined() && moment->dtype() == value.dtype() && moment->shape() == value.shape() &&
            moment->backend() == value.backend() && !moment->requires_grad(), "Invalid complex optimizer moment descriptor");
        for (const auto z : moment->complex_data()) require(std::isfinite(z.real()) && std::isfinite(z.imag()) &&
            (moment != &state_.second_moment_double[i] || (z.real() >= 0. && z.imag() >= 0.)), "Invalid complex optimizer moment value");
      }
      for (const auto z : value.complex_data()) require(std::isfinite(z.real()) && std::isfinite(z.imag()), "Non-finite complex parameter");
      wide_gradients[i] = Tensor(value.node()->gradient).detach();
      for (const auto z : wide_gradients[i].complex_data()) {
        require(std::isfinite(z.real()) && std::isfinite(z.imag()), "Non-finite complex gradient");
        norm_squared += std::norm(z);
      }
    } else if (value.dtype() == DType::Float64) {
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
                  state_.first_moment_double.size() == count && state_.second_moment_double.size() == count,
              "Double optimizer state storage mismatch: " + parameter.name);
      for (const auto* moment : {&state_.first_moment_double[i], &state_.second_moment_double[i]}) {
        require(moment->defined() && moment->dtype() == DType::Float64 && moment->shape() == value.shape() &&
                    moment->backend() == value.backend() && !moment->requires_grad(), "Invalid Double optimizer moment: " + parameter.name);
        for (double x : moment->double_data()) require(std::isfinite(x) &&
            (moment != &state_.second_moment_double[i] || x >= 0.), "Invalid Double optimizer moment value: " + parameter.name);
      }
      for (double x : value.double_data()) require(std::isfinite(x), "Non-finite parameter: " + parameter.name);
      wide_gradients[i] = Tensor(value.node()->gradient).detach();
      for (double x : wide_gradients[i].double_data()) {
        require(std::isfinite(x), "Non-finite parameter gradient: " + parameter.name);
        norm_squared += x * x;
      }
    } else if (low) {
      require(state_.first_moment[i].empty() && state_.second_moment[i].empty() &&
          (state_.first_moment_double.empty() || !state_.first_moment_double[i].defined()) &&
          (state_.second_moment_double.empty() || !state_.second_moment_double[i].defined()) &&
          state_.first_moment_float.size()==count && state_.second_moment_float.size()==count &&
          state_.sgd_buffer_initialized.size()==count && state_.sgd_buffer_initialized[i]<=1 &&
          (name_=="sgd" || state_.sgd_buffer_initialized[i]==0), "Invalid low-format optimizer state storage: " + parameter.name);
      for(const auto* moment:{&state_.first_moment_float[i],&state_.second_moment_float[i]}) {
        require(moment->defined() && moment->dtype()==DType::Float32 && moment->shape()==value.shape() &&
            moment->backend()==value.backend() && !moment->requires_grad(), "Invalid low-format optimizer moment: " + parameter.name);
        for(float x:moment->data())require(std::isfinite(x) && (moment!=&state_.second_moment_float[i] || x>=0.f), "Invalid low-format optimizer moment value: " + parameter.name);
      }
      for(float x:value.data())require(std::isfinite(x), "Non-finite low-format parameter: " + parameter.name);
      wide_gradients[i]=Tensor(value.node()->gradient).detach();
      for(float x:wide_gradients[i].data()) {require(std::isfinite(x), "Non-finite parameter gradient: " + parameter.name);norm_squared+=double(x)*x;}
    } else {
      require((state_.first_moment_double.empty() || !state_.first_moment_double[i].defined()) &&
                  (state_.second_moment_double.empty() || !state_.second_moment_double[i].defined()), "Unexpected Double optimizer state");
      gradients[i] = value.grad();
      require(gradients[i].size() == value.numel() && state_.first_moment[i].size() == value.numel() &&
                  state_.second_moment[i].size() == value.numel(), "Optimizer state shape mismatch: " + parameter.name);
      for (float x : gradients[i]) {
        require(std::isfinite(x), "Non-finite parameter gradient: " + parameter.name);
        norm_squared += static_cast<double>(x) * x;
      }
    }
  }
  const double norm = std::sqrt(norm_squared);
  const double scale = settings_.clip_norm > 0 ? std::min(1.0, settings_.clip_norm / (norm + 1e-6)) : 1.;
  TrainingState next = state_;
  ++next.step;
  const optimizer_internal::StepParameters step{rate, scale, settings_.weight_decay, settings_.momentum,
      settings_.beta1, settings_.beta2, settings_.epsilon,
      1 - std::pow(settings_.beta1, static_cast<double>(next.step)),
      1 - std::pow(settings_.beta2, static_cast<double>(next.step))};
  // Prepare every update before mutating parameters or advancing any state.
  for (size_t i = 0; i < count; ++i) {
    const auto& parameter = model.parameters()[i];
    if (state_.uses_resident_float32(parameter.value)) {
      DeviceScope owner(parameter.value.backend(), parameter.value.device());
      auto result = optimizer_internal::resident_float32_step(name_, parameter.value.detach(), wide_gradients[i],
          state_.first_moment_float[i], state_.second_moment_float[i], step);
      validate_resident_float32(result.value, false, parameter.name);
      validate_resident_float32(result.first_moment, false, parameter.name);
      if (name_ != "sgd") validate_resident_float32(result.second_moment, true, parameter.name);
      proposed_wide[i] = std::move(result.value);
      next.first_moment_float[i] = std::move(result.first_moment);
      next.second_moment_float[i] = std::move(result.second_moment);
    } else if (is_complex(parameter.value.dtype())) {
      auto result = optimizer_internal::complex_step(name_, parameter.value.detach(), wide_gradients[i],
          state_.first_moment_double[i], state_.second_moment_double[i], step, !state_.sgd_buffer_initialized[i]);
      for (const auto* tensor : {&result.value, &result.first_moment, &result.second_moment})
        for (const auto z : tensor->complex_data()) require(std::isfinite(z.real()) && std::isfinite(z.imag()), "Non-finite complex optimizer update");
      proposed_wide[i] = std::move(result.value);
      next.first_moment_double[i] = std::move(result.first_moment);
      next.second_moment_double[i] = std::move(result.second_moment);
      if (name_ == "sgd") next.sgd_buffer_initialized[i] = 1;
    } else if (parameter.value.dtype() == DType::Float64) {
      auto result = optimizer_internal::double_step(name_, parameter.value.detach(), wide_gradients[i],
          state_.first_moment_double[i], state_.second_moment_double[i], step);
      for (const auto* tensor : {&result.value, &result.first_moment, &result.second_moment})
        for (double x : tensor->double_data()) require(std::isfinite(x), "Non-finite Double optimizer update: " + parameter.name);
      proposed_wide[i] = std::move(result.value);
      next.first_moment_double[i] = std::move(result.first_moment);
      next.second_moment_double[i] = std::move(result.second_moment);
    } else if (is_low_precision_float(parameter.value.dtype())) {
      auto result=optimizer_internal::low_step(name_,parameter.value.detach(),wide_gradients[i],
          state_.first_moment_float[i],state_.second_moment_float[i],step,!state_.sgd_buffer_initialized[i]);
      for(const auto* tensor:{&result.value_float32,&result.value,&result.first_moment,&result.second_moment})
        for(float x:tensor->data())require(std::isfinite(x), "Non-finite low-format optimizer update: " + parameter.name);
      proposed_wide[i]=std::move(result.value);
      next.first_moment_float[i]=std::move(result.first_moment);
      next.second_moment_float[i]=std::move(result.second_moment);
      if(name_=="sgd")next.sgd_buffer_initialized[i]=1;
    } else {
      auto& values = proposed_narrow[i];
      values = parameter.value.data();
      if (name_ == "sgd") optimizer_internal::sgd_step(values, gradients[i], next.first_moment[i], step);
      else if (name_ == "adam") optimizer_internal::adam_step(values, gradients[i], next.first_moment[i], next.second_moment[i], step);
      else optimizer_internal::adamw_step(values, gradients[i], next.first_moment[i], next.second_moment[i], step);
      for (float x : values) require(std::isfinite(x), "Non-finite optimizer update: " + parameter.name);
    }
  }
  for (size_t i = 0; i < count; ++i) {
    auto& value = model.parameters()[i].value;
    if (proposed_wide[i].defined()) {
      DeviceScope owner(value.backend(), value.device());
      dispatch_tensor_into("copy", value, {proposed_wide[i]});
      ++value.node()->value->version;
    } else value.assign(proposed_narrow[i]);
  }
  state_ = std::move(next);
  return norm;
}

}  // namespace elman
