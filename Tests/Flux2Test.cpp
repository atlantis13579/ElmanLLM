#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Core/TensorDispatch.h"
#include "Core/LocalFunction.h"
#include "Inference/Diffusion/Flux2Sampler.h"
#include "Models/Diffusion/Flux2Text.h"
#include "Operators/flux2_ops.h"

namespace {
using namespace elman;
using namespace elman::flux2;
Backend backend;

Tensor data(Shape shape, uint32_t salt, float scale = 1.f, float offset = 0.f, DType dtype = DType::Float32,
            Backend location = backend) {
  std::vector<float> values(shape_element_count(shape));
  for (size_t i = 0; i < values.size(); ++i) {
    uint32_t x = uint32_t(i) + salt * 100003u;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x = ((x >> 16) ^ x) * 0x45d9f3bu;
    x ^= x >> 16;
    values[i] = (int(x % 257) - 128) / 128.f * scale + offset;
  }
  return Tensor::from_data(shape, values, location, false, dtype);
}

void close(float actual, float expected, float tolerance, const char* operation) {
  if (!(std::abs(actual - expected) <= tolerance * (1 + std::abs(expected)))) {
    throw std::runtime_error(std::string(operation) + ": " + std::to_string(actual) + " vs " +
                             std::to_string(expected));
  }
}
void compare(const Tensor& actual, const Tensor& expected, float tolerance, const char* operation) {
  if (actual.shape() != expected.shape() || actual.dtype() != expected.dtype() || actual.backend() != backend) {
    throw std::runtime_error(std::string(operation) + " descriptor mismatch");
  }
  std::vector<float> a = actual.data(), b = expected.data();
  for (size_t i = 0; i < a.size(); ++i) {
    close(a[i], b[i], tolerance, operation);
  }
}

void primitives(DType type) {
  const float tolerance = type == DType::Float32 ? 3e-5f : type == DType::BFloat16 ? .006f : .001f;
  Tensor ids = concat({text_positions(3, backend), image_positions(1, 2, 2, backend)}, 1);
  std::vector<int32_t> coords = ids.int_data();
  Tensor all_ids = concat({ids, add(ids, Tensor::from_int_data({}, {1}, backend))}, 0);
  Tensor x = data({2, 3, 5, 128}, 6, 1, 0, type);
  std::vector<float> y = apply_rotary(x, rotary_table(all_ids)).data(), input = x.data();
  for (size_t i = 0; i < y.size(); ++i) {
    const int pair = (i % 32) / 2, axis = (i % 128) / 32, token = (i / 128) % 5;
    const double angle = (coords[token * 4 + axis] + int(i / (3 * 5 * 128))) * std::pow(2000., -pair / 16.);
    close(y[i], float(input[i] * std::cos(angle) + (i % 2 ? 1 : -1) * input[i ^ 1] * std::sin(angle)), tolerance,
          "RoPE");
  }

  Tensor moments = data({2, 64, 4, 6}, 2, 1, 0, type);
  Tensor mean = data({128}, 3, .25f), var = data({128}, 4, .25f, .5f);
  Tensor packed = pack_latents(moments, mean, var);
  if (packed.shape() != Shape{2, 128, 2, 3}) {
    throw std::runtime_error("Invalid packed latent shape");
  }
  std::vector<float> values = moments.data(), m = mean.data(), v = var.data(), z = packed.data();
  for (int b = 0; b < 2; ++b) {
    for (int c = 0; c < 128; ++c) {
      for (int h = 0; h < 2; ++h) {
        for (int w = 0; w < 3; ++w) {
          const int i = ((b * 64 + c / 4) * 4 + 2 * h + (c / 2) % 2) * 6 + 2 * w + c % 2;
          close(z[((b * 128 + c) * 2 + h) * 3 + w], (values[i] - m[c]) / std::sqrt(v[c] + 1e-4f), tolerance, "pack");
        }
      }
    }
  }
  compare(unpack_latents(packed, mean, var), slice(moments, 1, 0, 32), tolerance, "unpack");

  // Both sequence lengths end in partial GPU tiles. Compare to independently
  // materialized FP32 logits so the CPU kernel is also checked against SDPA.
  Tensor q = data({2, 2, 17, 128}, 4, 2, 0, type), k = data({2, 2, 37, 128}, 5, 2, 0, type);
  Tensor value = data(k.shape(), 6, 1, 0, type);
  Tensor scores = div(matmul_transposed(cast(q, DType::Float32), cast(k, DType::Float32)), std::sqrt(128.));
  Tensor expected = cast(matmul(softmax(scores, -1), cast(value, DType::Float32)), type);
  compare(attention(q, k, value), expected, tolerance, "SDPA");

  Tensor features = data({2, 3, 131}, 72, 2, 0, type), shift = data({2, 1, 131}, 73, 1, 0, type),
       scale = data(shift.shape(), 74, 1, 0, type);
  Tensor work = cast(features, DType::Float32);
  Tensor centered = sub(work, elman::mean(work, -1, true));
  Tensor norm = cast(mul(centered, rsqrt(add(elman::mean(mul(centered, centered), -1, true), 1e-6))), type);
  Tensor mod = add(mul(norm, add(scale, 1.0)), shift);
  compare(modulate(features, shift, scale), mod, tolerance, "adaptive LayerNorm");
}

StreamWeights stream(int salt) {
  return {data({384, 128}, salt, .09f),     data({128}, salt + 1, .25f, 1),   data({128}, salt + 2, .25f, 1),
          data({128, 128}, salt + 3, .09f), data({128, 128}, salt + 4, .09f), data({128, 64}, salt + 5, .09f)};
}

void golden(const Tensor& output, const std::vector<float>& values, const char* name) {
  std::vector<float> y = output.data();
  size_t k = 0;
  for (int row = 0; row < 7; ++row) {
    for (int d : {0, 17, 46, 94, 127}) {
      close(y[row * 128 + d], values.at(k++), 1e-4f, name);
    }
  }
}

void blocks() {
  DoubleBlock dual{stream(10), stream(20)};
  SingleBlock single{data({512, 128}, 30, .09f), data({128, 192}, 31, .09f), data({128}, 32, .25f, 1),
                     data({128}, 33, .25f, 1)};
  Tensor txt = data({1, 2, 128}, 40), img = data({1, 5, 128}, 41);
  Tensor im = data({1, 5, 768}, 42, .8f), tm = data({1, 1, 768}, 43, .8f), sm = data({1, 7, 384}, 44, .8f);
  Tensor table = rotary_table(
      concat({text_positions(2, backend), image_positions(1, 2, 1, backend), image_positions(1, 3, 0, backend)}, 1));
  ReferenceKV dk, sk;
  std::pair<Tensor, Tensor> dual_output = dual.forward(img, txt, im, tm, {table, 2, 2, nullptr, &dk});
  Tensor& a = dual_output.first;
  Tensor& b = dual_output.second;
  Tensor h = concat({b, a}, 1);
  // Upstream DoubleStreamBlock and SingleStreamBlock at 50fe516, loaded with
  // the exact weights above, FP32, including the reference-only attention rows.
  golden(h, {.83786821f,  .95305598f,  1.05583763f, .20499669f,  -.13763049f, -.05888770f, -.70501852f,
             .45194510f,  .38787770f,  -.61828715f, -.55418396f, -.19055693f, .15910256f,  .26730454f,
             -.26622325f, .50185096f,  -.14830787f, -.77336711f, -.08997259f, -.38426873f, .26936108f,
             .27895564f,  -.65939558f, .58194804f,  .14251103f,  -.30757350f, .50041974f,  -.05830752f,
             -.01409120f, .67561632f,  .03257236f,  1.21311271f, .24534360f,  -.61199558f, .31547800f},
         "dual");
  Tensor result = single.forward(h, sm, {table, 2, 2, nullptr, &sk});
  golden(result, {.70468765f,  .79106975f,  1.03458452f, -.01659277f, -.10533967f, -.01132936f, -.68442726f,
                  .24906994f,  .37740690f,  -.58718187f, -.50231957f, -.31670773f, .15393150f,  .25572693f,
                  -.45572716f, .58771276f,  -.17560937f, -.79769808f, -.13116145f, -.35683146f, .02382120f,
                  .43303254f,  -.63390821f, .48701465f,  .21936914f,  -.28336352f, .44595477f,  -.15907216f,
                  -.00561579f, .62764698f,  -.17107378f, 1.17165005f, .31806418f,  -.61318916f, .45245385f},
         "single");
  Tensor (*const live)(const Tensor&) = [](const Tensor& x) { return concat({slice(x, 1, 0, 2), slice(x, 1, 4, 7)}, 1); };
  Tensor rt = live(table);
  std::pair<Tensor, Tensor> next = dual.forward(slice(img, 1, 2, 5), txt, slice(im, 1, 2, 5), tm, {rt, 2, 0, &dk});
  Tensor& img2 = next.first;
  Tensor& txt2 = next.second;
  compare(single.forward(concat({txt2, img2}, 1), live(sm), {rt, 2, 0, &sk}), live(result), 1e-4f, "cached blocks");
}

void text_encoder(const std::filesystem::path& directory, bool qwen) {
  std::filesystem::create_directories(directory);
  const std::string prefix = qwen ? "model." : "language_model.model.";
  std::ofstream config = std::ofstream(directory / "config.json");
  if (!qwen) {
    config << R"({"model_type":"mistral3","text_config":)";
  }
  config << R"({"hidden_size":8,"intermediate_size":12,"num_hidden_layers":36,"num_attention_heads":2,)"
         << R"("num_key_value_heads":1,"head_dim":4,"vocab_size":7,)"
         << (qwen ? R"("model_type":"qwen3","rms_norm_eps":1e-6,"rope_theta":1e6})"
                  : R"("model_type":"mistral","rms_norm_eps":1e-5,"rope_theta":1e9}})");
  config.close();
  std::map<std::string, Tensor> weights;
  LocalFunction put{[&](const std::string& name, Shape shape, int salt, float scale, float offset = 0) {
    weights.emplace(prefix + name, data(shape, salt, scale, offset, DType::Float32, Backend::CPU));
  }};
  put("embed_tokens.weight", {7, 8}, 57, 1);
  for (int layer = 0; layer < (qwen ? 27 : 30); ++layer) {
    const std::string p = "layers." + std::to_string(layer) + ".";
    const int s = 100 + 11 * layer;
    put(p + "input_layernorm.weight", {8}, s, .2f, 1);
    put(p + "self_attn.q_proj.weight", {8, 8}, s + 1, .2f);
    put(p + "self_attn.k_proj.weight", {4, 8}, s + 2, .2f);
    put(p + "self_attn.v_proj.weight", {4, 8}, s + 3, .2f);
    if (qwen) {
      put(p + "self_attn.q_norm.weight", {4}, s + 4, .2f, 1);
      put(p + "self_attn.k_norm.weight", {4}, s + 5, .2f, 1);
    }
    put(p + "self_attn.o_proj.weight", {8, 8}, s + 6, .2f);
    put(p + "post_attention_layernorm.weight", {8}, s + 7, .2f, 1);
    put(p + "mlp.gate_proj.weight", {12, 8}, s + 8, .2f);
    put(p + "mlp.up_proj.weight", {12, 8}, s + 9, .2f);
    put(p + "mlp.down_proj.weight", {8, 12}, s + 10, .2f);
  }
  save_safetensors(directory / "model.safetensors", weights);
  Tensor ids = Tensor::from_int64_data({1, 4}, {3, 6, 1, 5}, backend);
  Tensor mask = Tensor::from_int_data({1, 4}, {1, 1, 0, 0}, backend);
  Tensor context = TextEncoder(directory, DType::Float32).encode(ids, mask);
  if (context.shape() != Shape{1, 4, 24}) {
    throw std::runtime_error("Wrong text conditioning layout");
  }
  const std::vector<float> y = context.data();
  // PyTorch causal grouped SDPA with key padding, including the padded queries.
  const std::vector<float> gold =
      qwen ? std::vector<float>{-.646925f,  -.352894f, -1.527340f, -.364192f,  -.585452f, -.116839f, .441286f,
                                -2.095474f, .129832f,  .171712f,   .679863f,   -.289085f, -.177232f, 1.118732f,
                                -.464984f,  -.230504f, -.097124f,  -1.388429f, .096252f,  -.247688f}
           : std::vector<float>{-.450200f,  -.401181f, -1.395391f, -.773472f,  -.266014f, .185164f,  .291914f,
                                -1.846236f, -.162180f, .485336f,   1.013126f,  -.344480f, -.165720f, .793530f,
                                -.175310f,  .005852f,  -.195334f,  -1.252641f, -.332383f, .057820f};
  size_t i = 0;
  for (int row = 0; row < 4; ++row) {
    for (int d : {0, 7, 10, 16, 23}) {
      close(y[row * 24 + d], gold[i++], 2e-4f, "text conditioning");
    }
  }
}

void sampling() {
  for (std::pair<int, float> c : {std::pair<int, float>{64, .87536883f}, {4300, .90957677f}, {4301, .76579124f}}) {
    std::vector<float> ts = flow_schedule(4, c.first);
    close(ts[2], c.second, 2e-7f, "schedule");
    Tensor x = data({2, 3, 128}, 12), start = x, v = data(x.shape(), 16);
    for (int i = 0; i < 4; ++i) {
      x = flow_step(x, v, ts[i], ts[i + 1]);
    }
    compare(x, sub(start, v), 2e-6f, "Euler");
  }
  Tensor cond = data({1, 3, 128}, 97), unc = data(cond.shape(), 98);
  compare(guided_velocity(unc, cond, 4), sub(mul(cond, 4), mul(unc, 3)), 2e-6f, "CFG");
  std::vector<float> emb = timestep_embedding(Tensor::from_data({2}, {.125f, 4.f}, backend)).data();
  for (int j = 0; j < 256; ++j) {
    const double phase = 125 * std::pow(10000., -double(j % 128) / 128);
    close(emb[j], float(j < 128 ? std::cos(phase) : std::sin(phase)), 1e-5f, "time");
  }
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc != 3) {
      return 1;
    }
    backend = parse_backend(argv[1]);
    if (!backend_available(backend)) {
      return 77;
    }
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
      primitives(dtype);
    }
    blocks();
    sampling();
    for (bool qwen : {true, false}) {
      text_encoder(std::filesystem::path(argv[2]) / (qwen ? "qwen" : "mistral"), qwen);
    }
    std::cout << "FLUX.2 operator checks passed on " << backend_name(backend) << '\n';
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
