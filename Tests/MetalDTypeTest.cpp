#include "Core/Tensor.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
using namespace elman;
constexpr Backend backend = Backend::Metal;
size_t checks = 0;

void check(bool valid, const std::string& message) {
  ++checks;
  if (!valid) throw std::runtime_error(message);
}

void near(const std::vector<float>& actual, const std::vector<float>& expected,
          float tolerance, const std::string& label) {
  check(actual.size() == expected.size(), label + " element count");
  for (size_t i = 0; i < actual.size(); ++i) {
    check(std::isfinite(actual[i]) && std::isfinite(expected[i]) &&
              std::abs(actual[i] - expected[i]) <= tolerance * (1 + std::abs(expected[i])),
          label + " at " + std::to_string(i));
  }
}

void parity(const Tensor& actual, const Tensor& expected, float tolerance, const char* label) {
  check(actual.backend() == backend && actual.dtype() == expected.dtype() && actual.shape() == expected.shape(),
        std::string(label) + " descriptor");
  near(actual.data(), expected.data(), tolerance, label);
}

void exact(const Tensor& actual, const Tensor& expected, const char* label) {
  check(actual.backend() == backend && actual.dtype() == expected.dtype() && actual.shape() == expected.shape(),
        std::string(label) + " descriptor");
  check(actual.raw_data() == expected.raw_data(), label);
}

void unsupported(const std::function<void()>& operation) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::invalid_argument& error) {
    rejected = std::string(error.what()).find("Metal does not support") != std::string::npos;
  }
  check(rejected, "Wide floating arithmetic must explicitly report Metal dtype support");
}

void narrow_floats() {
  for (DType dtype : {DType::Float16, DType::BFloat16}) {
    const float tolerance = dtype == DType::Float16 ? .003f : .02f;
    std::vector<float> values(35);
    for (size_t i = 0; i < values.size(); ++i) values[i] = (int(i % 13) - 6) / 8.f;
    Tensor cpu = Tensor::from_data({5, 7}, values, Backend::CPU, true, dtype);
    Tensor gpu = Tensor::from_data({5, 7}, values, backend, true, dtype);
    exact(gpu, cpu, "Narrow floating upload");
    parity(add(gpu, gpu), add(cpu, cpu), 0, "Narrow floating add");
    parity(mul(gpu, gpu), mul(cpu, cpu), 0, "Narrow floating multiply");
    parity(sum(gpu, -1), sum(cpu, -1), tolerance, "Narrow floating reduction");
    parity(matmul(gpu, transpose(gpu, 0, 1)), matmul(cpu, transpose(cpu, 0, 1)), tolerance, "Narrow floating matmul");
    parity(softmax(gpu, -1), softmax(cpu, -1), tolerance, "Narrow floating softmax");
    parity(cast(gpu, DType::Float32), cast(cpu, DType::Float32), 0, "Narrow floating decode");
    sum_all(mul(gpu, gpu)).backward();
    sum_all(mul(cpu, cpu)).backward();
    near(gpu.grad(), cpu.grad(), tolerance, "Narrow floating backward");
    check(gpu.dtype() == dtype, "Backward changed parameter storage dtype");
  }
}

void low_precision_floats() {
  for (DType dtype : all_dtypes()) {
    if (!is_low_precision_float(dtype)) continue;
    // Raw code round trips include NaNs, signed zeros and odd packed tails.
    std::vector<uint8_t> codes(dtype_storage_bytes(dtype, 257));
    for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(i);
    Tensor storage = Tensor::from_raw({257}, dtype, codes, backend);
    check(storage.raw_data() == codes, "Low precision raw encoding round trip");
    Tensor cpu = Tensor::from_data({2, 3}, {1, 2, 4, 2, 1, 2}, Backend::CPU, false, dtype);
    Tensor gpu = cpu.to(backend);
    parity(cast(gpu, DType::Float32), cast(cpu, DType::Float32), 0, "Low precision decode");
    exact(cast(cast(gpu, DType::Float32), dtype), cpu, "Low precision encode");
    parity(add(gpu, gpu), add(cpu, cpu), 0, "Low precision add");
    parity(mul(gpu, gpu), mul(cpu, cpu), 0, "Low precision multiply");
    Tensor ones = Tensor::from_data({2, 2}, {1, 1, 1, 1}, Backend::CPU, false, dtype);
    Tensor device = ones.to(backend);
    parity(matmul(device, device), matmul(ones, ones), 0, "Low precision matmul");
    parity(sum(device, -1), sum(ones, -1), 0, "Low precision reduction");
    Tensor host_leaf = Tensor::from_data({5}, {1, 2, 1, 2, 1}, Backend::CPU, true, dtype);
    Tensor device_leaf = Tensor::from_data({5}, {1, 2, 1, 2, 1}, backend, true, dtype);
    sum_all(mul(host_leaf, host_leaf)).backward();
    sum_all(mul(device_leaf, device_leaf)).backward();
    near(device_leaf.grad(), host_leaf.grad(), 0, "Low precision backward");
  }
}

void integers() {
  for (DType dtype : all_dtypes()) {
    if (!is_arithmetic_integer(dtype)) continue;
    Tensor cpu = Tensor::from_int_data({2, 3}, {1, 0, 1, 0, 1, 1}, Backend::CPU, dtype);
    Tensor gpu = cpu.to(backend);
    exact(add(gpu, gpu), add(cpu, cpu), "Integer addition and wrapping");
    exact(mul(gpu, gpu), mul(cpu, cpu), "Integer multiplication");
    exact(sum(gpu, -1), sum(cpu, -1), "Integer sum and promotion");
    exact(max(gpu, -1), max(cpu, -1), "Integer maximum");
    exact(transpose(gpu, 0, 1), transpose(cpu, 0, 1), "Packed integer layout");
  }
  // Values beyond the FP64 integer mantissa must never be converted through a float.
  Tensor cpu = Tensor::from_int64_data({2, 2}, {9007199254740993LL, -9007199254740993LL, INT64_MAX - 1, INT64_MIN});
  Tensor gpu = cpu.to(backend);
  Tensor one = Tensor::from_int64_data({}, {1});
  exact(add(gpu, one.to(backend)), add(cpu, one), "Exact signed 64-bit arithmetic");
  exact(min(gpu, -1), min(cpu, -1), "Exact signed 64-bit ordering");
  const std::vector<uint64_t> values = {0, uint64_t(1) << 63, UINT64_MAX, 9007199254740993ULL};
  std::vector<uint8_t> bytes(values.size() * sizeof(uint64_t));
  std::memcpy(bytes.data(), values.data(), bytes.size());
  cpu = Tensor::from_raw({2, 2}, DType::UInt64, bytes);
  gpu = cpu.to(backend);
  exact(max(gpu, -1), max(cpu, -1), "Exact unsigned 64-bit ordering");
}

void wide_storage() {
  const uint64_t patterns[] = {0, 0x8000000000000000ULL, 0x7ff8000000000731ULL, 0x7ff0000000000000ULL,
                             0x3ff0000000000001ULL, 1, 0xfff0000000000000ULL, 0x8000000000000001ULL};
  std::vector<uint8_t> bytes(sizeof(patterns));
  std::memcpy(bytes.data(), patterns, bytes.size());
  for (DType dtype : {DType::Float64, DType::Complex128}) {
    const Shape shape = dtype == DType::Float64 ? Shape{2, 4} : Shape{2, 2};
    Tensor cpu = Tensor::from_raw(shape, dtype, bytes);
    Tensor gpu = cpu.to(backend);
    exact(gpu, cpu, "Wide floating raw storage");
    exact(transpose(gpu, 0, 1), transpose(cpu, 0, 1), "Wide floating transpose");
    exact(slice(gpu, 1, 0, 1), slice(cpu, 1, 0, 1), "Wide floating slice");
    exact(concat({gpu, gpu}, 0), concat({cpu, cpu}, 0), "Wide floating concatenation");
    unsupported([&] { add(gpu, gpu); });
    unsupported([&] { mul(gpu, gpu); });
    unsupported([&] { sum_all(gpu); });
    unsupported([&] { cast(gpu, DType::Float32); });
    check(gpu.raw_data() == bytes, "Rejected arithmetic changed wide storage");
    if (dtype == DType::Complex128) {
      exact(real(gpu), real(cpu), "Wide complex real component");
      exact(imag(gpu), imag(cpu), "Wide complex imaginary component");
      exact(conj(gpu), conj(cpu), "Wide complex conjugate bits");
      exact(complex(real(gpu), imag(gpu)), cpu, "Wide complex reconstruction");
    }
  }
  Tensor matrix = Tensor::from_double_data({2, 2}, {1, 2, 3, 4}, backend);
  unsupported([&] { matmul(matrix, matrix); });
  unsupported([&] { Tensor::uniform({8}, 0, 1, 7, backend, false, DType::Float64); });
}

void random_values() {
  constexpr int count = 4096;
  for (DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
    // Exercise both padded Philox draws and exact multiples of four.
    for (int size : {0, 1, 2, 3, 4, 5, 7, 8, 9}) {
      Tensor small = Tensor::uniform({size}, 1, 1.00001, 73, backend, false, dtype);
      exact(small, Tensor::uniform({size}, 1, 1.00001, 73, backend, false, dtype), "Small uniform reproducibility");
      for (float value : small.data())
        check(value >= 1 && value < 1.00001, "Small uniform escaped its narrow interval");
      Tensor normal = Tensor::normal({size}, 0, 1, 89, backend, false, dtype);
      exact(normal, Tensor::normal({size}, 0, 1, 89, backend, false, dtype), "Small normal reproducibility");
      Tensor x = Tensor::ones({size}, backend, true, dtype);
      Tensor dropped = dropout(x, .5, 91);
      exact(dropped, dropout(x, .5, 91), "Small dropout reproducibility");
      for (float value : dropped.data()) check(value == 0 || value == 2, "Small dropout scaling");
      if (size) {
        sum_all(dropped).backward();
        near(x.grad(), dropped.data(), 0, "Small dropout backward mask");
      }
    }
    Tensor uniform = Tensor::uniform({count}, -1, 1, 73, backend, false, dtype);
    exact(uniform, Tensor::uniform({count}, -1, 1, 73, backend, false, dtype), "Metal uniform reproducibility");
    check(uniform.raw_data() != Tensor::uniform({count}, -1, 1, 74, backend, false, dtype).raw_data(), "Uniform seed ignored");
    double mean = 0;
    const std::vector<float> values = uniform.data();
    for (size_t i = 0; i < values.size(); ++i) {
      const float value = values[i];
      check(std::isfinite(value) && value >= -1 && value < 1,
            "Metal uniform half-open interval for " + std::string(dtype_name(dtype)) + " at " +
                std::to_string(i) + ": " + std::to_string(value));
      mean += value / count;
    }
    check(std::abs(mean) < .06, "Metal uniform mean");
    Tensor normal = Tensor::normal({count}, 0, 1, 89, backend, false, dtype);
    exact(normal, Tensor::normal({count}, 0, 1, 89, backend, false, dtype), "Metal normal reproducibility");
    check(normal.raw_data() != Tensor::normal({count}, 0, 1, 90, backend, false, dtype).raw_data(), "Normal seed ignored");
    double square = 0;
    mean = 0;
    for (float value : normal.data()) {
      check(std::isfinite(value), "Metal normal finite output");
      mean += value / count;
      square += double(value) * value / count;
    }
    check(std::abs(mean) < .08 && std::abs(square - 1) < .12, "Metal normal moments");
    Tensor x = Tensor::ones({count}, backend, true, dtype);
    Tensor dropped = dropout(x, .5, 91);
    exact(dropped, dropout(x, .5, 91), "Metal dropout reproducibility");
    check(dropped.raw_data() != dropout(x, .5, 92).raw_data(), "Dropout seed ignored");
    size_t kept = 0;
    for (float value : dropped.data()) {
      check(value == 0 || value == 2, "Metal dropout inverted scaling");
      kept += value != 0;
    }
    check(std::abs(double(kept) / count - .5) < .05, "Metal dropout probability");
    sum_all(dropped).backward();
    near(x.grad(), dropped.data(), 0, "Metal dropout backward reuses forward mask");
    exact(dropout(x, 0., 91), x, "Metal zero dropout");
    exact(dropout(x, .5, 91, false), x, "Metal evaluation dropout");
    exact(dropout(x, 1., 91), Tensor::zeros({count}, backend, false, dtype), "Metal full dropout");
    x.zero_grad();
    sum_all(dropout(x, 1., 91)).backward();
    near(x.grad(), std::vector<float>(count, 0), 0, "Metal full dropout gradient");
    parity(Tensor::normal({7}, 2, 0, 73, backend, false, dtype),
           Tensor::from_data({7}, std::vector<float>(7, 2), Backend::CPU, false, dtype), 0, "Degenerate normal");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 2) throw std::invalid_argument("Expected a Metal dtype suite");
    if (!backend_available(backend)) return 77;
    const std::string suite = argv[1];
    if (suite == "float16") narrow_floats();
    else if (suite == "float8") low_precision_floats();
    else if (suite == "integer") integers();
    else if (suite == "wide_storage") wide_storage();
    else if (suite == "random") random_values();
    else throw std::invalid_argument("Unknown Metal dtype suite: " + suite);
    std::cout << "Metal " << suite << ": " << checks << " checks passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
