#include "Backends/CudaGemm.h"
#include "Backends/AttentionGemm.h"
#include "Backends/CudaStream.h"
#include "Core/Execution.h"
#include "Core/Tensor.h"
#include "Kernals/attributes.h"
#include "Kernals/matmul_params.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace elman {
void launch_matmul_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_linear_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
void launch_addmm_cuda(uint32_t*, const uint32_t*, const uint32_t*, const uint32_t*, const KernelParams&);
}
using namespace elman;
namespace {
void check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}
struct Allocation {
  void* pointer = nullptr;
  explicit Allocation(size_t bytes) { check(cudaMalloc(&pointer, std::max<size_t>(bytes, 16))); }
  ~Allocation() { cudaFree(pointer); }
  Allocation(const Allocation&) = delete;
  uint32_t* words() { return static_cast<uint32_t*>(pointer); }
};
struct Case {
  int m, n, k;
  bool ta = false, tb = false;
  GemmOperation operation = GemmOperation::Matmul;
  int bias = -1;
  float alpha = 1, beta = 1;
};
void input_shape(KernelParams& p, KernelInput input, std::vector<int> shape) {
  const int slot = KernelParams::input_slot(input);
  p.v[slot] = int(shape.size());
  int stride = 1;
  for (int i = int(shape.size()) - 1; i >= 0; --i) {
    p.v[slot + KernelParams::kInputShapeOffset + i] = shape[i];
    p.v[slot + KernelParams::kInputStrideOffset + i] = stride;
    stride *= shape[i];
  }
}
KernelParams params(const Case& s, DType dtype) {
  KernelParams p;
  p.v[KernelParams::kOutputCountSlot] = s.m * s.n;
  p.v[KernelParams::kOutputRankSlot] = 2;
  p.v[KernelParams::kOutputShapeSlot] = s.m;
  p.v[KernelParams::kOutputShapeSlot + 1] = s.n;
  input_shape(p, KernelInput::A, s.ta ? std::vector<int>{s.k, s.m} : std::vector<int>{s.m, s.k});
  input_shape(p, KernelInput::B, s.tb ? std::vector<int>{s.n, s.k} : std::vector<int>{s.k, s.n});
  if (s.bias < 0) p.v[KernelParams::input_slot(KernelInput::C)] = -1;
  else if (s.bias == 0) input_shape(p, KernelInput::C, {});
  else if (s.bias == 1) input_shape(p, KernelInput::C, {s.n});
  else if (s.bias == 2) input_shape(p, KernelInput::C, {s.m, 1});
  else input_shape(p, KernelInput::C, {s.m, s.n});
  for (int i = 0; i < 3; ++i) p.v[KernelParams::kInputDTypeSlot + i] = int(dtype);
  p.v[KernelParams::kOutputDTypeSlot] = int(dtype);
  if (s.operation == GemmOperation::Matmul) {
    p.v[KernelParams::kAttributeSlot + operator_attr::matmul::kTransposeA] = s.ta;
    p.v[KernelParams::kAttributeSlot + operator_attr::matmul::kTransposeB] = s.tb;
  } else if (s.operation == GemmOperation::Addmm) {
    std::memcpy(&p.v[KernelParams::kAttributeSlot + operator_attr::addmm::kAlpha], &s.alpha, sizeof(float));
    std::memcpy(&p.v[KernelParams::kAttributeSlot + operator_attr::addmm::kBeta], &s.beta, sizeof(float));
  }
  return p;
}
void launch_case(const Case& s, uint32_t* out, const uint32_t* a, const uint32_t* b, const uint32_t* c, const KernelParams& p) {
  if (s.operation == GemmOperation::Linear) launch_linear_cuda(out, a, b, c, p);
  else if (s.operation == GemmOperation::Addmm) launch_addmm_cuda(out, a, b, c, p);
  else launch_matmul_cuda(out, a, b, c, p);
}
std::vector<uint8_t> encode(std::vector<float>& data, DType dtype) {
  std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, data.size()));
  for (size_t i = 0; i < data.size(); ++i) {
    if (dtype == DType::Float32) std::memcpy(bytes.data() + 4 * i, &data[i], 4);
    else {
      uint16_t bits;
      if (dtype == DType::Float16) {
        __half h = __float2half_rn(data[i]);
        data[i] = __half2float(h); std::memcpy(&bits, &h, 2);
      } else {
        __nv_bfloat16 h = __float2bfloat16_rn(data[i]);
        data[i] = __bfloat162float(h); std::memcpy(&bits, &h, 2);
      }
      std::memcpy(bytes.data() + 2 * i, &bits, 2);
    }
  }
  return bytes;
}
float decode(const uint8_t* data, DType dtype) {
  if (dtype == DType::Float32) { float v; std::memcpy(&v, data, 4); return v; }
  if (dtype == DType::Float16) { __half v; std::memcpy(&v, data, 2); return __half2float(v); }
  __nv_bfloat16 v; std::memcpy(&v, data, 2); return __bfloat162float(v);
}
void near(float actual, double expected, DType dtype, size_t index) {
  if (std::isnan(expected)) { if (std::isnan(actual)) return; }
  else if (std::isinf(expected)) { if (actual == expected) return; }
  else {
    const double tolerance = dtype == DType::Float32 ? 3e-5 : dtype == DType::Float16 ? .0015 : .012;
    if (std::isfinite(actual) && std::abs(actual - expected) <= tolerance * (1 + std::abs(expected))) return;
  }
  throw std::runtime_error("GEMM mismatch at " + std::to_string(index) + ": " + std::to_string(actual) + " vs " + std::to_string(expected));
}
void run_case(const Case& s, DType dtype, bool benchmark = false, bool ignored_nan = false) {
  const KernelParams p = params(s, dtype);
  std::mt19937 random(20261126);
  std::uniform_real_distribution<float> distribution(-.25f, .25f);
  std::vector<float> a(size_t(s.m) * s.k), b(size_t(s.n) * s.k);
  std::vector<float> c(s.bias <= 0 ? 1 : s.bias == 1 ? s.n : s.bias == 2 ? s.m : size_t(s.m) * s.n);
  for (auto* values : {&a, &b, &c}) for (float& value : *values) value = distribution(random);
  if (ignored_nan) {
    if (s.alpha == 0) std::fill(a.begin(), a.end(), std::numeric_limits<float>::quiet_NaN());
    if (s.beta == 0) std::fill(c.begin(), c.end(), std::numeric_limits<float>::quiet_NaN());
  }
  const auto ab = encode(a, dtype), bb = encode(b, dtype), cb = encode(c, dtype);
  const size_t width = dtype_storage_bytes(dtype, 1), count = size_t(s.m) * s.n;
  // One-element offsets deliberately violate vector-load alignment. Guard words
  // check both the leading and trailing boundary of the output allocation.
  const size_t guard = benchmark ? 256 : width;
  Allocation da(ab.size() + guard), db(bb.size() + guard), dc(cb.size()), dout(count * width + 2 * guard);
  auto* ap = static_cast<uint8_t*>(da.pointer) + guard;
  auto* bp = static_cast<uint8_t*>(db.pointer) + guard;
  auto* op = static_cast<uint8_t*>(dout.pointer) + guard;
  // Pageable synchronous uploads may return before their default-stream copy
  // completes. Order uploads with the kernels on the nonblocking test stream.
  if (!ab.empty()) check(cudaMemcpyAsync(ap, ab.data(), ab.size(), cudaMemcpyHostToDevice, current_cuda_stream()));
  if (!bb.empty()) check(cudaMemcpyAsync(bp, bb.data(), bb.size(), cudaMemcpyHostToDevice, current_cuda_stream()));
  check(cudaMemcpyAsync(dc.pointer, cb.data(), cb.size(), cudaMemcpyHostToDevice, current_cuda_stream()));
  check(cudaMemsetAsync(dout.pointer, 0x5a, count * width + 2 * guard, current_cuda_stream()));
  auto call = [&] { launch_case(s, reinterpret_cast<uint32_t*>(op), reinterpret_cast<const uint32_t*>(ap),
                                reinterpret_cast<const uint32_t*>(bp), dc.words(), p); };
  call();
  check(cudaStreamSynchronize(current_cuda_stream()));
  std::vector<uint8_t> output(count * width + 2 * guard);
  check(cudaMemcpy(output.data(), dout.pointer, output.size(), cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < guard; ++i)
    if (output[i] != 0x5a || output[count * width + guard + i] != 0x5a) throw std::runtime_error("GEMM output guard overwritten");
  const size_t step = benchmark ? std::max<size_t>(1, count / 257) : 1;
  for (size_t index = 0; index < count; index += step) {
    const int r = int(index / s.n), n = int(index % s.n);
    double value = 0;
    if (s.alpha != 0) for (int k = 0; k < s.k; ++k)
      value += double(a[s.ta ? size_t(k) * s.m + r : size_t(r) * s.k + k]) *
                      b[s.tb ? size_t(n) * s.k + k : size_t(k) * s.n + n];
    value *= s.alpha;
    if (s.bias >= 0 && s.beta != 0)
      value += double(s.beta) * c[s.bias == 0 ? 0 : s.bias == 1 ? n : s.bias == 2 ? r : index];
    try { near(decode(output.data() + index * width + guard, dtype), value, dtype, index); }
    catch (const std::exception& error) {
      throw std::runtime_error(std::string(dtype_name(dtype)) + " " + std::to_string(s.m) + "x" + std::to_string(s.n) +
          "x" + std::to_string(s.k) + (s.ta ? " T" : " N") + (s.tb ? "T: " : "N: ") + error.what());
    }
  }
  if (!benchmark) return;
  for (int i = 0; i < 10; ++i) call();
  check(cudaStreamSynchronize(current_cuda_stream()));
  // Graph replay removes Windows host launch gaps from short-kernel timings.
  // All builds use the same graph size for a shape. Bound the replay count for
  // multi-billion-FLOP products while retaining enough work for event timing.
  const bool large = 2. * s.m * s.n * s.k > 2e9;
  const int graph_operations = large ? 4 : 32, graph_replays = large ? 2 : 16;
  cudaGraph_t graph;
  cudaGraphExec_t executable;
  check(cudaStreamBeginCapture(current_cuda_stream(), cudaStreamCaptureModeThreadLocal));
  for (int i = 0; i < graph_operations; ++i) call();
  check(cudaStreamEndCapture(current_cuda_stream(), &graph));
  check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
  // Let desktop GPUs leave their idle clock state before collecting samples.
  const auto warmup_end = std::chrono::steady_clock::now() + std::chrono::milliseconds(150);
  do {
    for (int i = 0; i < graph_replays; ++i) check(cudaGraphLaunch(executable, current_cuda_stream()));
    check(cudaStreamSynchronize(current_cuda_stream()));
  } while (std::chrono::steady_clock::now() < warmup_end);
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start)); check(cudaEventCreate(&stop));
  std::vector<float> samples;
  for (int sample = 0; sample < 7; ++sample) {
    check(cudaEventRecord(start, current_cuda_stream()));
    for (int i = 0; i < graph_replays; ++i) check(cudaGraphLaunch(executable, current_cuda_stream()));
    check(cudaEventRecord(stop, current_cuda_stream())); check(cudaEventSynchronize(stop));
    float ms = 0; check(cudaEventElapsedTime(&ms, start, stop));
    samples.push_back(ms * 1000.f / (graph_operations * graph_replays));
  }
  std::sort(samples.begin(), samples.end());
  const float us = samples[samples.size() / 2];
  std::cout << dtype_name(dtype) << ',' << (s.operation == GemmOperation::Linear ? "linear" : "matmul") << ','
            << s.m << ',' << s.n << ',' << s.k << ',' << (s.ta ? 'T' : 'N') << (s.tb ? 'T' : 'N')
            << ',' << std::fixed << std::setprecision(3) << us << ',' << 2. * s.m * s.n * s.k / (us * 1000.) << std::endl;
  check(cudaEventDestroy(start)); check(cudaEventDestroy(stop));
  check(cudaGraphExecDestroy(executable)); check(cudaGraphDestroy(graph));
}
void broadcast_and_gradient(DType dtype) {
  std::vector<float> av(2 * 1 * 7 * 11), bv(1 * 3 * 11 * 9);
  for (size_t i = 0; i < av.size(); ++i) av[i] = float(int(i % 23) - 11) * .03125f;
  for (size_t i = 0; i < bv.size(); ++i) bv[i] = float(int(i % 17) - 8) * .03125f;
  Tensor a = Tensor::from_data({2, 1, 7, 11}, av, Backend::CUDA, true, dtype);
  Tensor b = Tensor::from_data({1, 3, 11, 9}, bv, Backend::CUDA, true, dtype);
  Tensor product = matmul(a, b);
  const auto values = product.data();
  for (int x = 0; x < 2; ++x) for (int y = 0; y < 3; ++y)
    for (int r = 0; r < 7; ++r) for (int n = 0; n < 9; ++n) {
      double value = 0;
      for (int k = 0; k < 11; ++k) value += double(av[(x * 7 + r) * 11 + k]) * bv[(y * 11 + k) * 9 + n];
      const size_t i = ((x * 3 + y) * 7 + r) * 9 + n;
      near(values[i], value, dtype, i);
    }
  sum_all(product).backward();
  const auto ga = a.grad(), gb = b.grad();
  for (int x = 0; x < 2; ++x) for (int r = 0; r < 7; ++r) for (int k = 0; k < 11; ++k) {
    double value = 0;
    for (int y = 0; y < 3; ++y) for (int n = 0; n < 9; ++n) value += bv[(y * 11 + k) * 9 + n];
    near(ga[(x * 7 + r) * 11 + k], value, dtype, k);
  }
  for (int y = 0; y < 3; ++y) for (int k = 0; k < 11; ++k) for (int n = 0; n < 9; ++n) {
    double value = 0;
    for (int x = 0; x < 2; ++x) for (int r = 0; r < 7; ++r) value += av[(x * 7 + r) * 11 + k];
    near(gb[(y * 11 + k) * 9 + n], value, dtype, k);
  }
}
void large_batch() {
  Tensor a = Tensor::ones({65537, 1, 1}, Backend::CUDA);
  Tensor b = Tensor::from_data({1, 1}, {.25f}, Backend::CUDA);
  for (float value : matmul(a, b).data())
    if (value != .25f) throw std::runtime_error("GEMM batch grid boundary failed");
}
void fp32_precision() {
  for (int m : {128, 896}) {
    const int n = m, k = 256;
    std::vector<float> values(m * k);
    for (int i = 0; i < m * k; ++i) values[i] = i % 2 ? -1.f : 1.f + 0x1p-14f;
    Tensor a = Tensor::from_data({m, k}, values, Backend::CUDA);
    Tensor b = Tensor::ones({k, n}, Backend::CUDA);
    for (float value : matmul(a, b).data())
      if (value != k / 2 * 0x1p-14f) throw std::runtime_error("FP32 GEMM lost low mantissa bits");
    std::fill(values.begin(), values.end(), 1e20f);
    a = Tensor::from_data({m, k}, values, Backend::CUDA);
    b = Tensor::from_data({k, n}, std::vector<float>(k * n, 1e-20f), Backend::CUDA);
    for (float value : matmul(a, b).data()) near(value, double(1e20f) * double(1e-20f) * k, DType::Float32, 0);
  }
}
void attention_layout(DType dtype, bool pointer_batch, bool transpose, int m = 17, int n = 19, int k = 33) {
  const int batches = 6, groups = 3;
  const int leading = (transpose ? k : n) + 7;
  const int astride = (m + 5) * k, bstride = (transpose ? n : k) * leading, ostride = (m + 3) * n;
  std::vector<float> a(batches * astride), b(batches / groups * bstride), initial(batches * ostride, -77.f);
  for (size_t i = 0; i < a.size(); ++i) a[i] = float(int(i * 7 % 31) - 15) * .03125f;
  for (size_t i = 0; i < b.size(); ++i) b[i] = float(int(i * 13 % 29) - 14) * .03125f;
  const auto ab = encode(a, dtype), bb = encode(b, dtype), ob = encode(initial, dtype);
  const size_t width = dtype_storage_bytes(dtype, 1);
  Allocation da(ab.size()), db(bb.size()), dout(ob.size()), arrays(3 * batches * sizeof(void*));
  check(cudaMemcpyAsync(da.pointer, ab.data(), ab.size(), cudaMemcpyHostToDevice, current_cuda_stream()));
  check(cudaMemcpyAsync(db.pointer, bb.data(), bb.size(), cudaMemcpyHostToDevice, current_cuda_stream()));
  check(cudaMemcpyAsync(dout.pointer, ob.data(), ob.size(), cudaMemcpyHostToDevice, current_cuda_stream()));
  AttentionGemm s{dtype, m, n, k, batches, groups, astride, bstride, ostride, transpose, false, leading};
  std::vector<void*> pointers(3 * batches);
  if (pointer_batch) {
    for (int batch = 0; batch < batches; ++batch) {
      // Deliberately non-monotonic addresses distinguish pointer batches from strides.
      const int target = (batch * 5) % batches;
      pointers[batch] = static_cast<char*>(db.pointer) + size_t(target / groups) * bstride * width;
      pointers[batches + batch] = static_cast<char*>(da.pointer) + size_t(target) * astride * width;
      pointers[2 * batches + batch] = static_cast<char*>(dout.pointer) + size_t(target) * ostride * width;
    }
    check(cudaMemcpyAsync(arrays.pointer, pointers.data(), pointers.size() * sizeof(void*), cudaMemcpyHostToDevice, current_cuda_stream()));
    attention_gemm_cuda_pointers(arrays.pointer, s);
  } else attention_gemm_cuda(dout.words(), da.words(), db.words(), arrays.pointer, s);
  check(cudaStreamSynchronize(current_cuda_stream()));
  std::vector<uint8_t> actual(ob.size());
  check(cudaMemcpy(actual.data(), dout.pointer, actual.size(), cudaMemcpyDeviceToHost));
  for (int batch = 0; batch < batches; ++batch) for (int r = 0; r < m + 3; ++r) for (int c = 0; c < n; ++c) {
    double expected = -77;
    if (r < m) {
      expected = 0;
      for (int inner = 0; inner < k; ++inner)
        expected += double(a[batch * astride + r * k + inner]) *
          b[(batch / groups) * bstride + (transpose ? c * leading + inner : inner * leading + c)];
    }
    const size_t i = batch * ostride + r * n + c;
    near(decode(actual.data() + i * width, dtype), expected, dtype, i);
  }
}
} // namespace
int main(int argc, char** argv) {
  try {
    int devices = 0;
    if (cudaGetDeviceCount(&devices) != cudaSuccess || !devices) return 77;
    ExecutionStream stream(Backend::CUDA);
    StreamScope scope(stream);
    set_cuda_tf32_math(false, false);
    const bool large_benchmark = argc == 2 && std::string(argv[1]) == "--benchmark-fp32";
    const bool benchmark = large_benchmark || (argc == 2 && std::string(argv[1]) == "--benchmark");
    for (DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
      for (auto shape : {Case{1, 1, 1}, Case{1, 67, 113}, Case{4, 17, 65}, Case{7, 13, 19},
                         Case{16, 65, 33}, Case{33, 67, 31}, Case{65, 63, 97}, Case{257, 65, 17}, Case{32, 64, 64}})
        for (bool ta : {false, true}) for (bool tb : {false, true}) {
          shape.ta = ta; shape.tb = tb; run_case(shape, dtype);
        }
      for (int bias : {-1, 0, 1}) run_case({37, 71, 59, false, true, GemmOperation::Linear, bias}, dtype);
      for (int bias : {0, 1, 2, 3}) run_case({37, 71, 59, false, false, GemmOperation::Addmm, bias, .75f, -.5f}, dtype);
      run_case({7, 13, 0}, dtype);
      run_case({0, 13, 19}, dtype);
      run_case({7, 0, 19}, dtype);
      run_case({7, 13, 0, false, false, GemmOperation::Addmm, 1, .75f, .5f}, dtype);
      run_case({7, 13, 19, false, false, GemmOperation::Addmm, 1, 0, 1}, dtype, false, true);
      run_case({7, 13, 19, false, false, GemmOperation::Addmm, 1, 1, 0}, dtype, false, true);
      broadcast_and_gradient(dtype);
      for (bool pointers : {false, true}) for (bool transpose : {false, true}) attention_layout(dtype, pointers, transpose);
    }
    large_batch();
    fp32_precision();
    for (auto shape : {Case{64, 64, 64}, Case{96, 96, 65}, Case{129, 130, 97}, Case{257, 129, 17},
                       Case{769, 769, 65}, Case{895, 893, 65}})
      for (bool ta : {false, true}) for (bool tb : {false, true}) {
        shape.ta = ta; shape.tb = tb; run_case(shape, DType::Float32);
      }
    run_case({895, 893, 65, false, true, GemmOperation::Linear, 1}, DType::Float32);
    for (int bias : {0, 1, 2, 3})
      run_case({895, 893, 65, false, false, GemmOperation::Addmm, bias, .75f, -.5f}, DType::Float32);
    for (bool pointers : {false, true}) for (bool transpose : {false, true}) {
      attention_layout(DType::Float32, pointers, transpose, 129, 130, 65);
      attention_layout(DType::Float32, pointers, transpose, 511, 255, 65);
    }
    std::cerr << "CUDA GEMM: transpose, tails, unaligned views, broadcast gradients, epilogues and empty/NaN cases passed\n";
    if (benchmark) {
      cudaDeviceProp device{}; check(cudaGetDeviceProperties(&device, 0));
      const auto versions = gemm_library_versions_cuda();
      std::cerr << device.name << "; cuBLAS=" << versions.blas << "; cuBLASLt=" << versions.blas_lt << "; TF32=off\n";
      std::cout << "dtype,operation,m,n,k,layout,median_us,gflops\n";
      if (large_benchmark) {
        for (auto shape : {Case{128, 512, 512}, Case{256, 512, 512}, Case{512, 512, 512},
                           Case{768, 768, 768}, Case{1024, 1024, 1024}, Case{1536, 1536, 1536}, Case{2048, 2048, 2048},
                           Case{128, 2048, 1024}, Case{2048, 128, 1024}, Case{256, 4096, 512},
                           Case{513, 509, 521}, Case{1025, 769, 257}})
          for (bool ta : {false, true}) for (bool tb : {false, true}) {
            shape.ta = ta; shape.tb = tb; run_case(shape, DType::Float32, true);
          }
        return 0;
      }
      for (DType dtype : {DType::Float32, DType::Float16, DType::BFloat16})
        for (auto s : {Case{1, 512, 512}, Case{4, 1024, 1024}, Case{16, 256, 256}, Case{64, 256, 256},
                       Case{128, 512, 512}, Case{256, 512, 512}, Case{512, 512, 512}, Case{37, 71, 59}}) {
          run_case(s, dtype, true);
          s.tb = true; s.operation = GemmOperation::Linear; s.bias = 1;
          run_case(s, dtype, true);
        }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "CUDA GEMM test failed: " << e.what() << '\n'; return 1;
  }
}
