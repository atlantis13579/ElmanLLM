#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "Backends/CudaGemm.h"
#include "Core/Tensor.h"
#include "Core/Graph.h"
#include "Core/Execution.h"

namespace {

struct Result {
  std::vector<float> output;
  std::vector<float> input_gradient;
  std::vector<float> weight_gradient;
};

Result run(bool matmul_tf32, bool convolution_tf32, float input_scale = 1.f) {
  elman::set_cuda_tf32_math(matmul_tf32, convolution_tf32);
  if (elman::cuda_conv_tf32_enabled() != convolution_tf32)
    throw std::runtime_error("Convolution math mode was not applied");
  std::vector<float> input(16 * 32 * 32), weight(16 * 16 * 3 * 3);
  for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i * 7 % 19) - 9) * .01f * input_scale;
  for (size_t i = 0; i < weight.size(); ++i) weight[i] = float(int(i * 11 % 23) - 11) * .01f;
  elman::Tensor x = elman::Tensor::from_data({1, 16, 32, 32}, input, elman::Backend::CUDA, true);
  elman::Tensor w = elman::Tensor::from_data({16, 16, 3, 3}, weight, elman::Backend::CUDA, true);
  elman::Tensor output = elman::conv2d(x, w, elman::Tensor{}, 1, 1);
  if (output.dtype() != elman::DType::Float32)
    throw std::runtime_error("Convolution output must remain Float32");
  Result result;
  result.output = output.data();
  elman::sum_all(output).backward();
  if (elman::Tensor(x.node()->gradient).dtype() != elman::DType::Float32 ||
      elman::Tensor(w.node()->gradient).dtype() != elman::DType::Float32)
    throw std::runtime_error("Convolution gradients must remain Float32");
  result.input_gradient = x.grad();
  result.weight_gradient = w.grad();
  return result;
}

void close(const std::vector<float>& reference, const std::vector<float>& actual, float tolerance = .02f) {
  if (reference.size() != actual.size() || reference.empty())
    throw std::runtime_error("Convolution output or gradient shape differs");
  for (size_t i = 0; i < reference.size(); ++i)
    if (!std::isfinite(actual[i]) || std::abs(reference[i] - actual[i]) > tolerance)
      throw std::runtime_error("Convolution math mode changed the result excessively");
}

void cached_stream_modes() {
  using namespace elman;
  std::vector<float> data(128 * 128);
  for (size_t i = 0; i < data.size(); ++i) data[i] = std::sin(float(i) * .013f) * .17f;
  Tensor x = Tensor::from_data({128, 128}, data, Backend::CUDA);
  Tensor w = Tensor::from_data({128, 128}, data, Backend::CUDA);
  Tensor bias = Tensor::zeros({128}, Backend::CUDA);
  ExecutionStream stream(Backend::CUDA);
  for (bool tf32 : {false, true, false, true}) {
    set_cuda_tf32_math(tf32, !tf32);
    const auto reference = linear(x, w, bias).data();
    const auto matrix_reference = matmul(x, w).data();
    StreamScope scope(stream);
    Tensor output = linear(x, w, bias);
    if (output.dtype() != DType::Float32 || w.dtype() != DType::Float32 || output.data() != reference || matmul(x, w).data() != matrix_reference)
      throw std::runtime_error("Cached CUDA stream did not refresh its FP32 math mode");
  }
  for (DType dtype : {DType::Float16, DType::BFloat16, DType::Float64}) {
    Tensor a = cast(x, dtype), b = cast(w, dtype);
    set_cuda_tf32_math(false, false);
    const auto reference = matmul(a, b).data();
    set_cuda_tf32_math(true, true);
    Tensor actual = matmul(a, b);
    if (actual.dtype() != dtype || actual.data() != reference)
      throw std::runtime_error("TF32 policy affected a non-Float32 GEMM");
  }
  set_cuda_tf32_math(false, false);
}
}  // namespace

int main() {
  try {
    const Result reference = run(false, false);
    for (const auto [matmul, convolution] : {std::pair{false, true},
                                            std::pair{true, false},
                                            std::pair{true, true}}) {
      const Result result = run(matmul, convolution);
      close(reference.output, result.output);
      close(reference.input_gradient, result.input_gradient);
      close(reference.weight_gradient, result.weight_gradient);
    }
    cached_stream_modes();
    // TF32 retains the FP32 exponent range; converting operands to FP16 would overflow.
    const Result wide_reference = run(false, false, 1e7f), wide_tf32 = run(false, true, 1e7f);
    close(wide_reference.output, wide_tf32.output, 2000.f);
    close(wide_reference.input_gradient, wide_tf32.input_gradient);
    close(wide_reference.weight_gradient, wide_tf32.weight_gradient, 20000.f);
    std::cout << "CUDA FP32/TF32 matmul, convolution and cached stream math modes passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "CUDA math mode test failure: " << error.what() << '\n';
    return 1;
  }
}
