#pragma once

#include "NN/Module.h"

namespace elman::nn {

class Sequential final : public Module {
 public:
  Module& append(std::string name, std::unique_ptr<Module> child);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  std::vector<Module*> order_;
};

class Linear final : public Module {
 public:
  Linear(int inputs, int outputs, Backend backend, uint32_t seed = 1, bool bias = true);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  Tensor weight_, bias_;
};

class Conv2d final : public Module {
 public:
  Conv2d(int inputs, int outputs, int kernel, Backend backend, uint32_t seed = 1,
         int stride = 1, int padding = 0, bool bias = true);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  int stride_, padding_;
  Tensor weight_, bias_;
};

class GroupNorm final : public Module {
 public:
  GroupNorm(int groups, int channels, Backend backend, double epsilon = 1e-5);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  int groups_;
  double epsilon_;
  Tensor weight_, bias_;
};

class ReLU final : public Module {
  Tensor forward(const Tensor& input, uint32_t) override { return relu(input); }
};

class Dropout final : public Module {
 public:
  explicit Dropout(double probability);
 private:
  Tensor forward(const Tensor& input, uint32_t seed) override;
  double probability_;
};

}  // namespace elman::nn
