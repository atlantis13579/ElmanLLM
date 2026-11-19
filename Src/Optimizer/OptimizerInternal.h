#pragma once

#include <cstddef>
#include <vector>
#include "Core/Tensor.h"

namespace elman::optimizer_internal {

struct StepParameters {
  double rate;
  double gradient_scale;
  double weight_decay;
  double momentum;
  double beta1;
  double beta2;
  double epsilon;
  double correction1;
  double correction2;
};

struct DoubleStep {
  Tensor value;
  Tensor first_moment;
  Tensor second_moment;
};
DoubleStep double_step(const std::string& algorithm, const Tensor& values,
                       const Tensor& gradients, const Tensor& first_moment,
                       const Tensor& second_moment, const StepParameters& step);
DoubleStep complex_step(const std::string& algorithm, const Tensor& values, const Tensor& gradients,
                        const Tensor& first_moment, const Tensor& second_moment,
                        const StepParameters& step, bool first_step);
struct LowStep {
  Tensor value;
  Tensor first_moment;
  Tensor second_moment;
  Tensor value_float32;
};
LowStep low_step(const std::string& algorithm,const Tensor& values,const Tensor& gradients,
                const Tensor& first_moment,const Tensor& second_moment,const StepParameters& step,bool first_step);
DoubleStep resident_float32_step(const std::string& algorithm, const Tensor& values, const Tensor& gradients,
                                const Tensor& first_moment, const Tensor& second_moment, const StepParameters& step);

void sgd_step(std::vector<float>& values, const std::vector<float>& gradients,
              std::vector<float>& first_moment, const StepParameters& step);
void adam_step(std::vector<float>& values, const std::vector<float>& gradients,
               std::vector<float>& first_moment,
               std::vector<float>& second_moment, const StepParameters& step);
void adamw_step(std::vector<float>& values, const std::vector<float>& gradients,
                std::vector<float>& first_moment,
                std::vector<float>& second_moment, const StepParameters& step);

}  // namespace elman::optimizer_internal
