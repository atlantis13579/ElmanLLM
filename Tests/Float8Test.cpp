#include "Core/AllocationPool.h"
#include "Core/LocalFunction.h"
#include "Core/Execution.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"
#include "IO/Json.h"
#include "Inference/LLM/Sampling.h"
#include "Optimizer/Optimizer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
namespace kernals {
using namespace elman;
int checks = 0;
size_t elements = 0, nan_encodings = 0;
void check(bool valid, const std::string& message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
std::vector<uint8_t> bytes(const Json& spec) {
    const std::string text = spec.at("raw_hex").string();
    if (text.size() % 2) throw std::runtime_error("Odd hex length");
    int (*const digit)(char) = [](char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        throw std::runtime_error("Invalid hex digit");
    };
    std::vector<uint8_t> result(text.size() / 2);
    for (size_t i = 0; i < result.size(); ++i) result[i] = uint8_t(digit(text[2 * i]) * 16 + digit(text[2 * i + 1]));
    return result;
}
Shape shape(const Json& spec) {
    Shape result;
    for (const elman::Json& value : spec.at("shape").array()) result.push_back(value.integer());
    return result;
}
bool nan_code(uint64_t code, DType dtype) {
    switch (dtype) {
    case DType::Float32: return (code & 0x7fffffff) > 0x7f800000;
    case DType::Float64: return (code & 0x7fffffffffffffffull) > 0x7ff0000000000000ull;
    case DType::Float16: return (code & 0x7fff) > 0x7c00;
    case DType::BFloat16: return (code & 0x7fff) > 0x7f80;
    case DType::Float8E4M3FN: return (code & 127) == 127;
    case DType::Float8E5M2: return (code & 127) > 124;
    case DType::Float8E4M3FNUZ: case DType::Float8E5M2FNUZ: return code == 128;
    case DType::Float8E8M0FNU: return code == 255;
    default: return false;
    }
}
void compare(const Tensor& output, const Json& expected, const std::string& name) {
    const std::vector<uint8_t> wanted = bytes(expected), actual = output.raw_data();
    check(output.dtype() == DType(expected.at("dtype").integer()) && output.shape() == shape(expected), name + ": layout differs");
    check(actual.size() == wanted.size(), name + ": byte count differs");
    elements += output.numel();
    if (output.dtype() == DType::Float4E2M1FN) {
        for (size_t i = 0; i < actual.size(); ++i)
            if (actual[i] != wanted[i]) throw std::runtime_error(name + ": packed FP4 bits differ at byte " +
                std::to_string(i) + " (actual " + std::to_string(actual[i]) + ", expected " + std::to_string(wanted[i]) + ")");
        ++checks;
        return;
    }
    const size_t width = dtype_info(output.dtype()).bits / 8;
    for (size_t i = 0; i < output.numel(); ++i) {
        uint64_t a = 0, b = 0;
        std::memcpy(&a, actual.data() + i * width, width);
        std::memcpy(&b, wanted.data() + i * width, width);
        if (a == b) continue;
        if (nan_code(a, output.dtype()) && nan_code(b, output.dtype())) { ++nan_encodings; continue; }
        throw std::runtime_error(name + ": exact finite or signed-zero bits differ at " + std::to_string(i) +
                                 " (actual " + std::to_string(a) + ", expected " + std::to_string(b) + ")");
    }
    ++checks;
}
KernelParams descriptor(const Tensor& output, const Tensor& input) {
    KernelParams p;
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    p.v[KernelParams::input_slot(KernelInput::A)] = int(input.shape().size());
    p.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(input.dtype());
    int stride = 1;
    for (int d = int(input.shape().size()) - 1; d >= 0; --d) {
        p.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputShapeOffset + d] = input.shape()[d];
        p.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset + d] = stride;
        stride *= input.shape()[d];
    }
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    return p;
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Malformed low precision cast was accepted");
}
KernelParams pointwise_descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p = descriptor(output, inputs.at(0));
    for (size_t i = 1; i < inputs.size(); ++i) {
        const KernelInput input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void native(Backend backend) {
    const std::vector<float> values{0.f, -0.f, .5f, 1.f, -1.5f, 2.f, 4.f};
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor input = Tensor::from_data({7}, values, backend);
        Tensor output = cast(input, dtype);
        check(output.raw_data() == Tensor::from_data({7}, values, backend, false, dtype).raw_data(), "Upload and native cast differ");
        check(cast(output, DType::Float32).data() == output.data(), "Download and native decode differ");
        std::vector<uint8_t> codes(256);
        for (size_t i = 0; i < codes.size(); ++i) codes[i] = uint8_t(i);
        Tensor all = Tensor::from_raw({dtype == DType::Float4E2M1FN ? 512 : 256}, dtype, codes, backend);
        Tensor copy = Tensor::from_raw(all.shape(), dtype, std::vector<uint8_t>(256), backend);
        KernelParams p = descriptor(copy, all);
        backend_dispatch("cast", copy.node()->value, {all.node()->value}, p);
        check(copy.raw_data() == codes, "Same dtype raw cast changed NaN payloads or packed bits");
        KernelParams inplace = descriptor(all, all);
        backend_dispatch("cast", all.node()->value, {all.node()->value}, inplace);
        check(all.raw_data() == codes, "In-place raw cast changed bytes");
        KernelParams valid = descriptor(output, input);
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 6;
        rejects([&] { backend_dispatch("cast", output.node()->value, {input.node()->value}, bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 9;
        rejects([&] { backend_dispatch("cast", output.node()->value, {input.node()->value}, bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 2;
        rejects([&] { backend_dispatch("cast", output.node()->value, {input.node()->value}, bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B)] = 0;
        rejects([&] { backend_dispatch("cast", output.node()->value, {input.node()->value}, bad); });
        rejects([&] { backend_dispatch("cast", output.node()->value, {}, valid); });
        rejects([&] { cast(output, DType::Int32); });
        rejects([&] { add(output, Tensor::ones({7}, backend)); });
        const size_t shift = dtype == DType::Float4E2M1FN ? 2 : 1;
        Tensor first = contiguous_view(all, {6});
        Tensor shifted = contiguous_view(all, {6}, shift);
        const KernelParams overlap = descriptor(shifted, first);
        rejects([&] { backend_dispatch("cast", shifted.node()->value, {first.node()->value}, overlap); });
        check(all.raw_data() == codes, "Rejected overlapping cast modified storage");
        Tensor offset_input = contiguous_view(Tensor::from_data({8}, {99.f, 0.f, .5f, 1.f, 2.f, 4.f, 99.f, 99.f}, backend), {5}, 1);
        Tensor reference = cast(offset_input, dtype);
        const int root_count = dtype == DType::Float4E2M1FN ? 7 : 8;
        Tensor storage = Tensor::from_raw({root_count}, dtype, std::vector<uint8_t>(dtype_storage_bytes(dtype, size_t(root_count)), 0xa5), backend);
        Tensor target_view = contiguous_view(storage, {5}, 2);
        const KernelParams into_view = descriptor(target_view, offset_input);
        backend_dispatch("cast", target_view.node()->value, {offset_input.node()->value}, into_view);
        check(target_view.raw_data() == reference.raw_data(), "Cast into an offset storage view differs");
        const std::vector<uint8_t> written = storage.raw_data();
        check(written[0] == 0xa5 && (dtype == DType::Float4E2M1FN || (written[1] == 0xa5 && written[7] == 0xa5)),
              "Cast modified bytes outside the output view");
        if (dtype == DType::Float4E2M1FN) {
            check((written.back() & 0xf0) == 0, "Packed view conversion left a dirty padding nibble");
            rejects([&] { contiguous_view(storage, {3}, 2); });
        }
        check(cast(target_view, DType::Float32).data() == cast(reference, DType::Float32).data(), "Cast from an offset view differs");
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor encoded;
        {
            StreamScope scope(producer, asynchronous, true);
            encoded = cast(Tensor::from_data({7}, values, backend), DType::Float4E2M1FN);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            check(cast(encoded, DType::Float32).data() == values, "Packed conversion stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
Tensor evaluate(const Json& test, const std::vector<Tensor>& x) {
    const std::string op = test.at("op").string();
    if (op == "uniform" || op == "normal" || op == "dropout") {
        if (test.contains("replicate_count")) {
            const int repeats = test.at("replicate_count").integer();
            elman::Json::Object object = test.object(); object.erase("replicate_count");
            std::vector<float> values;
            Tensor output;
            for (int repeat = 0; repeat < repeats; ++repeat) {
                const int seed = int(uint32_t(test.at("seed").integer()) + uint32_t(repeat));
                object["seed"] = seed;
                elman::Json::Array attributes = object.at("attrs").array(); attributes[0] = seed; object["attrs"] = attributes;
                output = evaluate(Json(object), x);
                const std::vector<float> sample = output.data(); values.insert(values.end(), sample.begin(), sample.end());
            }
            Shape dimensions{repeats}; const Shape sample_shape = output.shape(); dimensions.insert(dimensions.end(), sample_shape.begin(), sample_shape.end());
            return Tensor::from_data(dimensions, values, output.backend(), false, output.dtype());
        }
        LocalFunction parameter{[&](const char* name) {
            const elman::Json::Array& words = test.at(name).array();
            const uint64_t bits = uint64_t(uint32_t(words.at(0).integer())) | (uint64_t(uint32_t(words.at(1).integer())) << 32);
            double value; std::memcpy(&value, &bits, sizeof(value)); return value;
        }};
        const Backend backend = x.empty() ? parse_backend(test.at("backend").string()) : x[0].backend();
        const DType dtype = op == "dropout" ? x.at(0).dtype() : DType(test.at("dtype").integer());
        Shape output_shape = op == "dropout" ? x.at(0).shape() : shape(test);
        if (test.bool_or("raw", false)) {
            std::vector<int> attributes;
            for (const elman::Json& v : test.at("attrs").array()) attributes.push_back(v.integer());
            return dispatch_tensor(op, output_shape, x, attributes, dtype, backend);
        }
        const uint32_t seed = uint32_t(test.at("seed").integer());
        if (op == "dropout") return dropout(x.at(0), parameter("p_bits"), seed, test.bool_or("training", true));
        if (op == "uniform") return Tensor::uniform(output_shape, parameter("first_bits"), parameter("second_bits"), seed, backend, false, dtype);
        return Tensor::normal(output_shape, parameter("first_bits"), parameter("second_bits"), seed, backend, false, dtype);
    }
    if (op == "sampling_scores" || op == "sampling_probabilities" || op == "sample_tokens") {
        LocalFunction scalar{[&](const std::string& name, double fallback) {
            if (!test.contains(name + "_bits")) return test.number_or(name, fallback);
            const elman::Json::Array& words = test.at(name + "_bits").array();
            const uint64_t bits = uint64_t(uint32_t(words.at(0).integer())) | (uint64_t(uint32_t(words.at(1).integer())) << 32);
            double value; std::memcpy(&value, &bits, sizeof(value)); return value;
        }};
        SamplingOptions options;
        options.temperature = scalar("temperature", 1.); options.top_p = scalar("top_p", 1.);
        options.repetition_penalty = scalar("repetition_penalty", 1.); options.top_k = test.int_or("top_k", 0);
        if (op == "sampling_scores") return sampling_scores(x.at(0), x.at(1), options);
        if (op == "sampling_probabilities") return softmax(sampling_scores(x.at(0), x.at(1), options), 1);
        SamplingRng rng(std::stoull(test.at("seed").string()));
        std::vector<int32_t> ids = sample_tokens(x.at(0), x.at(1), options, rng);
        return Tensor::from_int_data({int(ids.size())}, ids, x.at(0).backend());
    }
    if (op == "categorical_sample") {
        std::vector<int32_t> ids = categorical_sample(x.at(0), x.at(1).data());
        return Tensor::from_int_data({int(ids.size())}, ids, x.at(0).backend());
    }
    if (test.bool_or("raw", false)) {
        std::vector<int> attributes;
        for (const elman::Json& value : test.at("attrs").array()) attributes.push_back(value.integer());
        const Json& expected = test.at("expected");
        return dispatch_tensor(op, shape(expected), x, attributes, DType(expected.at("dtype").integer()));
    }
    if (op == "cast") return cast(x.at(0), DType(test.at("dtype").integer()));
    if (op == "cast_roundtrip") return cast(cast(x.at(0), DType(test.at("dtype").integer())), x.at(0).dtype());
    LocalFunction integers{[&](const char* name) { Shape values; for (const elman::Json& value : test.at(name).array()) values.push_back(value.integer()); return values; }};
    LocalFunction divisor{[&] {
        if (!test.contains("divisor_bits")) return test.number_or("divisor", 1.);
        const elman::Json::Array& words = test.at("divisor_bits").array();
        const uint64_t bits = uint64_t(uint32_t(words.at(0).integer())) | (uint64_t(uint32_t(words.at(1).integer())) << 32);
        double value; std::memcpy(&value, &bits, sizeof(value)); return value;
    }};
    if (op == "attention_softmax") return attention_softmax(x.at(0), x.at(1), divisor(), test.contains("axis") ? test.at("axis").integer() : -1);
    if (op == "tiled_attention") return tiled_attention(x.at(0), x.at(1), x.at(2), x.at(3), divisor(), test.contains("query_tile") ? test.at("query_tile").integer() : 32,
        test.bool_or("causal", false), test.contains("causal_offset") ? test.at("causal_offset").integer() : 0, AttentionTiling(test.contains("tiling") ? test.at("tiling").integer() : 0));
    if (op == "silu_mul") return silu_mul(x.at(0), x.at(1));
    if (op == "rotary_embedding") return rotary_embedding(x.at(0), x.at(1), x.at(2));
    if (op == "mse_loss") return mse_loss(x.at(0), x.at(1));
    if (op == "cross_entropy") return cross_entropy(x.at(0), x.at(1));
    if (op == "conv2d") return conv2d(x.at(0), x.at(1), x.size() > 2 ? x.at(2) : Tensor{}, test.at("stride").integer(), test.at("padding").integer());
    if (op == "max_pool2d") return max_pool2d(x.at(0), test.at("kernel").integer(), test.at("stride").integer(), test.at("padding").integer());
    if (op == "avg_pool2d") return avg_pool2d(x.at(0), test.at("kernel").integer(), test.at("stride").integer(), test.at("padding").integer());
    if (op == "reshape") return reshape(x.at(0), integers("shape"));
    if (op == "permute") return permute(x.at(0), integers("axes"));
    if (op == "transpose") return transpose(x.at(0), test.at("dim0").integer(), test.at("axis").integer());
    if (op == "slice") return slice(x.at(0), test.at("axis").integer(), test.at("start").integer(), test.at("stop").integer(), test.at("step").integer());
    if (op == "expand") return expand(x.at(0), integers("shape"));
    if (op == "squeeze") return test.contains("axis") ? squeeze(x.at(0), test.at("axis").integer()) : squeeze(x.at(0));
    if (op == "unsqueeze") return unsqueeze(x.at(0), test.at("axis").integer());
    if (op == "concat") return concat(x, test.at("axis").integer());
    if (op == "index_copy") return index_copy(x.at(0), test.at("axis").integer(), x.at(1), x.at(2));
    if (op == "contiguous_view") return contiguous_view(x.at(0), integers("shape"), size_t(test.at("offset").integer()));
    if (op == "triu") return triu(x.at(0), test.at("diagonal").integer());
    if (op == "tril") return tril(x.at(0), test.at("diagonal").integer());
    if (op == "argsort") return argsort(x.at(0), test.at("axis").integer(), test.at("descending").boolean());
    if (op == "sort") return sort(x.at(0), test.at("axis").integer(), test.at("descending").boolean()).first;
    if (op == "gather") return gather(x.at(0), test.at("axis").integer(), x.at(1));
    if (op == "index_select") return index_select(x.at(0), test.at("axis").integer(), x.at(1));
    if (op == "scatter") return scatter(x.at(0), test.at("axis").integer(), x.at(1), x.at(2));
    if (op == "repeat") {
        Shape reps;
        for (const elman::Json& value : test.at("reps").array()) reps.push_back(value.integer());
        return repeat(x.at(0), reps);
    }
    if (op == "where") return where(x.at(0), x.at(1), x.at(2));
    if (op == "layer_norm") return layer_norm(x.at(0), x.at(1), x.at(2), test.at("eps").number());
    if (op == "rms_norm") return rms_norm(x.at(0), x.at(1), test.at("eps").number(), RmsNormRounding(test.at("rounding").integer()));
    if (op == "batch_norm") return batch_norm(x.at(0), x.at(1), x.at(2), test.at("eps").number());
    if (op == "group_norm") return group_norm(x.at(0), test.at("groups").integer(), x.at(1), x.at(2), test.at("eps").number());
    if (op == "softmax") return softmax(x.at(0), test.at("axis").integer());
    if (op == "log_softmax") return log_softmax(x.at(0), test.at("axis").integer());
    if (op == "matmul") return matmul(x.at(0), x.at(1));
    if (op == "matmul_transposed") return matmul_transposed(x.at(0), x.at(1));
    if (op == "bmm") return bmm(x.at(0), x.at(1));
    if (op == "linear") return linear(x.at(0), x.at(1), x.size() > 2 ? x.at(2) : Tensor{});
    if (op == "addmm") return addmm(x.at(0), x.at(1), x.at(2), test.number_or("beta", 1.), test.number_or("alpha", 1.));
    if (op == "packed_linear") {
        std::vector<int> widths;
        for (const elman::Json& value : test.at("widths").array()) widths.push_back(value.integer());
        return packed_linear(x.at(0), x.at(1), widths, x.size() > 2 ? x.at(2) : Tensor{});
    }
    if (op == "sum_all") return sum_all(x.at(0));
    if (op == "mean_all") return mean_all(x.at(0));
    if (op == "cumsum") return cumsum(x.at(0), test.at("axis").integer());
#define ELMAN_LOW_REDUCTION(name) if (op == #name) return name(x.at(0), test.at("axis").integer(), test.at("keepdim").boolean())
    ELMAN_LOW_REDUCTION(sum); ELMAN_LOW_REDUCTION(mean); ELMAN_LOW_REDUCTION(prod);
    ELMAN_LOW_REDUCTION(max); ELMAN_LOW_REDUCTION(min); ELMAN_LOW_REDUCTION(amax); ELMAN_LOW_REDUCTION(amin);
#undef ELMAN_LOW_REDUCTION
#define ELMAN_LOW_UNARY(name) if (op == #name) return name(x.at(0))
    ELMAN_LOW_UNARY(sqrt); ELMAN_LOW_UNARY(rsqrt); ELMAN_LOW_UNARY(exp); ELMAN_LOW_UNARY(log);
    ELMAN_LOW_UNARY(sin); ELMAN_LOW_UNARY(cos); ELMAN_LOW_UNARY(abs); ELMAN_LOW_UNARY(sign);
    ELMAN_LOW_UNARY(relu); ELMAN_LOW_UNARY(silu); ELMAN_LOW_UNARY(gelu); ELMAN_LOW_UNARY(logical_not);
#undef ELMAN_LOW_UNARY
    if (op == "leaky_relu") return leaky_relu(x.at(0), test.number_or("slope", .01));
#define ELMAN_LOW_BINARY(name) if (op == #name) return name(x.at(0), x.at(1))
    ELMAN_LOW_BINARY(add); ELMAN_LOW_BINARY(sub); ELMAN_LOW_BINARY(mul); ELMAN_LOW_BINARY(div); ELMAN_LOW_BINARY(pow);
    ELMAN_LOW_BINARY(maximum); ELMAN_LOW_BINARY(minimum); ELMAN_LOW_BINARY(eq); ELMAN_LOW_BINARY(ne);
    ELMAN_LOW_BINARY(gt); ELMAN_LOW_BINARY(ge); ELMAN_LOW_BINARY(lt); ELMAN_LOW_BINARY(le);
    ELMAN_LOW_BINARY(logical_and); ELMAN_LOW_BINARY(logical_or);
#undef ELMAN_LOW_BINARY
#define ELMAN_LOW_SCALAR(name) if (op == #name "_scalar") return name(x.at(0), test.at("scalar").number())
    ELMAN_LOW_SCALAR(add); ELMAN_LOW_SCALAR(sub); ELMAN_LOW_SCALAR(mul); ELMAN_LOW_SCALAR(div); ELMAN_LOW_SCALAR(pow);
#undef ELMAN_LOW_SCALAR
    throw std::invalid_argument("Unknown low precision reference operation: " + op);
}
Json specification(const Tensor& value) {
    Json::Array dimensions;
    for (int dimension : value.shape()) dimensions.emplace_back(dimension);
    std::string raw; const char* digits = "0123456789abcdef";
    for (uint8_t code : value.raw_data()) { raw += digits[code >> 4]; raw += digits[code & 15]; }
    return Json::Object{{"shape", dimensions}, {"dtype", int(value.dtype())}, {"raw_hex", raw}};
}
void random_distribution(const Tensor& output, const Json& test, const Json& suite) {
    const Json& expected = test.at("expected");
    check(output.dtype() == DType(expected.at("dtype").integer()) && output.shape() == shape(expected), "Random distribution layout differs");
    if (is_compute_float(output.dtype())) {
        const Json& distribution = test.at("distribution");
        check(size_t(distribution.at("samples").integer()) == output.numel(), "Floating random sample count differs");
        std::vector<uint32_t> (*const ordered)(std::vector<float>) = [](std::vector<float> values) {
            std::vector<uint32_t> keys; keys.reserve(values.size());
            for (float value : values) {
                uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
                keys.push_back(std::isnan(value) ? UINT32_MAX : (bits & 0x80000000u) ? ~bits : bits ^ 0x80000000u);
            }
            std::sort(keys.begin(), keys.end()); return keys;
        };
        const std::vector<uint32_t> actual = ordered(output.data());
        const std::vector<uint32_t> wanted = ordered(Tensor::from_raw(shape(expected), output.dtype(), bytes(expected), Backend::CPU).data());
        size_t left = 0, right = 0; double distance = 0;
        while (left < actual.size() || right < wanted.size()) {
            const uint32_t key = std::min(left < actual.size() ? actual[left] : UINT32_MAX,
                                         right < wanted.size() ? wanted[right] : UINT32_MAX);
            while (left < actual.size() && actual[left] == key) ++left;
            while (right < wanted.size() && wanted[right] == key) ++right;
            distance = std::max(distance, std::abs(double(left) - double(right)) / double(actual.size()));
        }
        check(distance <= distribution.at("cdf_tolerance").number(), "Floating random CDF differs from PyTorch: " + std::to_string(distance));
        if (test.bool_or("require_reference_support", false))
            for (uint32_t key : actual) check(std::binary_search(wanted.begin(), wanted.end(), key), "Floating uniform sampling grid differs from PyTorch");
        elements += output.numel(); return;
    }
    const std::vector<uint8_t> raw = output.raw_data();
    std::vector<size_t> actual(257);
    for (size_t i = 0; i < output.numel(); ++i) {
        uint8_t code = output.dtype() == DType::Float4E2M1FN ? uint8_t((raw[i / 2] >> ((i % 2) * 4)) & 15) : raw[i];
        ++actual[nan_code(code, output.dtype()) ? 256 : code];
    }
    const Json& distribution = test.at("distribution");
    const elman::Json::Array& counts = distribution.at("counts").array();
    check(counts.size() == 257 && size_t(distribution.at("samples").integer()) == output.numel(), "Random histogram size differs");
    size_t left = 0, right = 0;
    double distance = 0;
    for (const elman::Json& index : suite.at("code_orders").at(std::to_string(int(output.dtype()))).array()) {
        const int code = index.integer(); left += actual.at(code); right += size_t(counts.at(code).integer());
        distance = std::max(distance, std::abs(double(left) - double(right)) / double(output.numel()));
    }
    check(left == output.numel() && right == output.numel(), "Random histogram has unexpected encodings");
    check(distance <= distribution.at("cdf_tolerance").number(), "Quantized random CDF differs from PyTorch: " + std::to_string(distance));
    if (output.dtype() == DType::Float4E2M1FN && output.numel() % 2)
        check((raw.back() & 0xf0) == 0, "Random packed padding is dirty");
    elements += output.numel();
}
std::pair<std::vector<float>, std::vector<float>> observed_dropout_scales(const Json& test, const Tensor& input) {
    const elman::Json::Array& words = test.at("p_bits").array();
    const uint64_t bits = uint64_t(uint32_t(words.at(0).integer())) | (uint64_t(uint32_t(words.at(1).integer())) << 32);
    double probability; std::memcpy(&probability, &bits, sizeof(probability));
    // E5M2 represents both an actual zero and an overflowed positive scale.
    // It makes the keep mask observable even for E8M0 and saturated FP4 output.
    const elman::DType control_dtype = is_low_precision_float(input.dtype()) ? DType::Float8E5M2 : input.dtype();
    Tensor ones = Tensor::ones(input.shape(), input.backend(), false, control_dtype);
    Tensor control = evaluate(test, {ones});
    const std::vector<float> values = control.data();
    LocalFunction scale_bits{[&](const char* key, float fallback) {
        if (!test.contains(key)) return fallback;
        const uint32_t bits = uint32_t(test.at(key).integer()); float value;
        std::memcpy(&value, &bits, sizeof(value)); return value;
    }};
    const float scale = scale_bits("reference_scale_bits", float(1. / (1. - probability)));
    const float backward_scale = scale_bits("reference_backward_scale_bits", scale);
    const float kept = Tensor::from_data({}, {scale}, input.backend(), false, control.dtype()).data().at(0);
    std::pair<std::vector<float>, std::vector<float>> scales{std::vector<float>(input.numel()), std::vector<float>(input.numel())}; size_t count = 0;
    for (size_t i = 0; i < scales.first.size(); ++i) {
        check((values[i] == 0 && !std::signbit(values[i])) || values[i] == kept, "Dropout control has an invalid quantized scale");
        if (values[i] == kept) { scales.first[i] = scale; scales.second[i] = backward_scale; ++count; }
    }
    if (!scales.first.empty()) check(std::abs(double(count) / double(scales.first.size()) - test.at("expected_keep_probability").number()) <= test.at("mask_probability_tolerance").number(), "Dropout keep probability differs from PyTorch");
    return scales;
}
Tensor conditioned_dropout(const Tensor& input, const std::vector<float>& scales) {
    std::vector<float> values = input.data();
    for (size_t i = 0; i < values.size(); ++i) values[i] *= scales[i];
    return Tensor::from_data(input.shape(), values, input.backend(), false, input.dtype());
}
void samplings(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN,DType::Float8E4M3FNUZ,DType::Float8E5M2,DType::Float8E5M2FNUZ,DType::Float8E8M0FNU,DType::Float4E2M1FN}) {
        Tensor storage=Tensor::ones({128},backend,false,dtype);
        Tensor weights=contiguous_view(storage,{2,3});
        Tensor cumulative=cumsum(cast(weights,DType::Float32),1);
        Tensor uniform=Tensor::from_data({2},{0.f,std::nextafter(1.f,0.f)},backend);
        Tensor output=Tensor::zeros({2},backend,false,DType::Int32);
        std::vector<Tensor> tensors{weights,cumulative,uniform};
        std::vector<BufferPtr> inputs{weights.node()->value,cumulative.node()->value,uniform.node()->value};
        KernelParams valid=pointwise_descriptor(output,tensors);
        LocalFunction execute{[&](const KernelParams& p){backend_dispatch("categorical",output.node()->value,inputs,p);}};
        execute(valid);check(output.int_data()==std::vector<int32_t>({0,2}),"Low precision categorical boundary differs");
        const std::vector<uint8_t> before=output.raw_data();
        std::vector<std::pair<int,int>> mutations{{KernelParams::kOutputCountSlot,0},{KernelParams::kOutputRankSlot,0},
            {KernelParams::kOutputRankSlot,9},{KernelParams::kOutputShapeSlot,3},{KernelParams::kOutputDTypeSlot,int(DType::Float32)},
            {KernelParams::input_slot(KernelInput::D),0},{KernelParams::kAttributeSlot,1},{KernelParams::kAttributeSlot+63,1}};
        for(int i=0;i<3;++i){
            const int slot=KernelParams::input_slot(KernelInput(i));
            mutations.insert(mutations.end(),{{slot,-1},{slot,9},{slot,0},{slot+KernelParams::kInputShapeOffset,3},
                {slot+KernelParams::kInputShapeOffset,-1},{slot+KernelParams::kInputStrideOffset,0},
                {KernelParams::input_dtype_slot(KernelInput(i)),int(DType::Float64)}});
            if(i<2)mutations.insert(mutations.end(),{{slot,1},{slot+KernelParams::kInputShapeOffset+1,0},
                {slot+KernelParams::kInputShapeOffset+1,INT32_MAX},{slot+KernelParams::kInputStrideOffset+1,2}});
            else mutations.push_back({slot,2});
        }
        for(std::pair<int, int> mutation:mutations){KernelParams bad=valid;bad.v[mutation.first]=mutation.second;rejects([&]{execute(bad);});
            check(output.raw_data()==before,"Rejected low precision categorical descriptor changed output");}
        rejects([&]{backend_dispatch("categorical",output.node()->value,{},valid);});
        std::vector<BufferPtr> extra=inputs;extra.push_back(weights.node()->value);
        rejects([&]{backend_dispatch("categorical",output.node()->value,extra,valid);});
        for(int i=0;i<3;++i){
            std::vector<BufferPtr> missing=inputs;missing[i].reset();rejects([&]{backend_dispatch("categorical",output.node()->value,missing,valid);});
            std::vector<Tensor> mixed=tensors;
            const elman::DType other=dtype==DType::Float8E4M3FN?DType::Float8E5M2:DType::Float8E4M3FN;
            mixed[i]=cast(mixed[i],other);
            if(i==0)mixed[1]=cast(cumulative,dtype);
            rejects([&]{dispatch_tensor_into("categorical",output,mixed);});
            if(backend==Backend::CUDA){mixed=tensors;mixed[i]=mixed[i].to(Backend::CPU);
                rejects([&]{dispatch_tensor_into("categorical",output,mixed);});}
            // The backing allocations have enough spare capacity for these
            // Int32 views, even if an overlap validation regression occurs.
            BufferPtr alias=backend_view(inputs[i],0,2);alias->dtype=DType::Int32;alias->bytes=8;
            const std::vector<uint8_t> original=tensors[i].raw_data();
            rejects([&]{backend_dispatch("categorical",alias,inputs,valid);});
            check(tensors[i].raw_data()==original,"Rejected low precision categorical alias changed input");
        }
        BufferPtr disjoint=backend_view(storage.node()->value,32,2);disjoint->dtype=DType::Int32;disjoint->bytes=8;
        backend_dispatch("categorical",disjoint,inputs,valid);
        check(backend_download_int(disjoint)==std::vector<int32_t>({0,2}),"Disjoint low precision categorical storage differs");
        Tensor empty=Tensor::zeros({0,3},backend,false,dtype);
        check(dispatch_tensor("categorical",{0},{empty,Tensor::zeros({0,3},backend),Tensor::zeros({0},backend)},{},DType::Int32).numel()==0,"Empty low precision categorical differs");
        const float invalid_weight=dtype==DType::Float8E8M0FNU?std::numeric_limits<float>::quiet_NaN():-1.f;
        Tensor invalid=Tensor::from_data({2,3},{1.f,invalid_weight,1.f,1.f,1.f,1.f},backend,false,dtype);
        check(dispatch_tensor("categorical",{2},{invalid,cumulative,uniform},{},DType::Int32).int_data()==std::vector<int32_t>({-1,2}),"Invalid low precision categorical row was not isolated");
        rejects([&]{categorical_sample(invalid,{0.f,.5f});});
        rejects([&]{categorical_sample(weights,{0.f});});
        rejects([&]{categorical_sample(weights,{0.f,1.f});});
        rejects([&]{categorical_sample_double(weights,{0.,.5});});
        Tensor differentiable=Tensor::ones({2,3},backend,true,dtype);
        Tensor history=Tensor::zeros({2,0},backend,false,DType::Int32);
        Tensor scores=sampling_scores(differentiable,history,{});
        check(scores.dtype()==DType::Float32&&!scores.requires_grad()&&scores.data()==std::vector<float>(6,1.f),"Low precision sampling scores did not detach into FP32");
        if(dtype==DType::Float4E2M1FN){
            Tensor dirty=Tensor::from_raw({1,3},dtype,{0x22,0xf2},backend);
            Tensor selected=dispatch_tensor("categorical",{1},{dirty,Tensor::from_data({1,3},{1.f,2.f,3.f},backend),Tensor::zeros({1},backend)},{},DType::Int32);
            check(selected.int_data()==std::vector<int32_t>{0}&&dirty.raw_data()==std::vector<uint8_t>({0x22,0xf2}),"Categorical read modified packed padding");
        }
        for(bool asynchronous:{false,true}){
            AllocationPool pool(backend);ExecutionStream producer(backend),consumer(backend);Tensor x;
            {StreamScope scope(producer,asynchronous,true);x=Tensor::ones({2,3},backend,true,dtype);}
            {StreamScope scope(consumer,asynchronous,true);
                check(categorical_sample(x,{0.f,.75f})==std::vector<int32_t>({0,2}),"Low precision categorical cross-stream dependency differs");
                SamplingRng first(42),second(42);
                check(sample_tokens(x,history,{},first)==sample_tokens(x,history,{},second),"Low precision seeded sampling replay differs");
                check(!sampling_scores(x,history,{}).requires_grad(),"Low precision sampling recorded a derivative");}
            producer.synchronize();consumer.synchronize();
        }
    }
}
void floating_randoms(Backend backend) {
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        for (bool normal : {false,true}) {
            const std::string op = normal ? "normal" : "uniform";
            const double first = normal ? 1./3 : -.1234567891234, second = normal ? 1.1234567891234 : 1.9876543219876;
            Tensor sample = normal ? Tensor::normal({4097},first,second,5721,backend,true,dtype) : Tensor::uniform({4097},first,second,5721,backend,true,dtype);
            std::vector<int> attributes(9);attributes[0]=5721;attributes[8]=1;
            std::memcpy(attributes.data()+4,&first,sizeof(first));std::memcpy(attributes.data()+6,&second,sizeof(second));
            Tensor raw=dispatch_tensor(op,sample.shape(),{},attributes,dtype,backend);
            check(raw.raw_data()==sample.raw_data(),"Public floating random scalar transport differs");
            check(raw.raw_data()==dispatch_tensor(op,sample.shape(),{},attributes,dtype,backend).raw_data(),"Floating random replay differs");
            attributes[1]=7;
            check(raw.raw_data()!=dispatch_tensor(op,sample.shape(),{},attributes,dtype,backend).raw_data(),"Floating random stream offsets collided");
            sample.backward(Tensor::ones(sample.shape(),backend,false,dtype));
            check(Tensor(sample.node()->gradient).raw_data()==Tensor::ones(sample.shape(),backend,false,dtype).raw_data(),"Floating random output is not a differentiable leaf");
            Tensor target=Tensor::zeros({7},backend,false,dtype);KernelParams valid=descriptor(target,target);
            valid.v[KernelParams::input_slot(KernelInput::A)]=-1;
            valid.v[KernelParams::kAttributeSlot]=5721;
            valid.v[KernelParams::kAttributeSlot+2]=0;valid.v[KernelParams::kAttributeSlot+3]=0x3f800000;
            LocalFunction execute{[&](const KernelParams& p){backend_dispatch(op,target.node()->value,{},p);}};
            execute(valid);const std::vector<uint8_t> unchanged=target.raw_data();
            for(std::pair<int, int> mutation:std::vector<std::pair<int,int>>{{KernelParams::kOutputCountSlot,6},{KernelParams::kOutputRankSlot,9},
                {KernelParams::kOutputShapeSlot,6},{KernelParams::kOutputDTypeSlot,int(DType::Int32)},
                {KernelParams::input_slot(KernelInput::A),0},{KernelParams::kAttributeSlot+8,2},{KernelParams::kAttributeSlot+9,1},
                {KernelParams::kAttributeSlot+(normal?3:2),int(normal?0xbf800000u:0x7fc00000u)}}){
                KernelParams bad=valid;bad.v[mutation.first]=mutation.second;rejects([&]{execute(bad);});
            }
            rejects([&]{backend_dispatch(op,target.node()->value,{target.node()->value},valid);});
            check(target.raw_data()==unchanged,"Rejected floating random descriptor changed output");
            Tensor empty=normal?Tensor::normal({2,0,3},0,1,5721,backend,false,dtype):Tensor::uniform({2,0,3},0,1,5721,backend,false,dtype);
            check(empty.shape()==Shape({2,0,3})&&empty.raw_data().empty(),"Floating random empty shape differs");
        }
        const double maximum=dtype==DType::Float16?65504.:dtype==DType::BFloat16?std::ldexp(255.,120):double(std::numeric_limits<float>::max());
        rejects([&]{Tensor::uniform({0},maximum,std::nextafter(maximum,std::numeric_limits<double>::infinity()),5721,backend,false,dtype);});
        rejects([&]{Tensor::uniform({7},-maximum,maximum,5721,backend,false,dtype);});
        rejects([&]{Tensor::normal({0},0,-1,5721,backend,false,dtype);});
        std::vector<float> constant=Tensor::normal({7},1e100,0,5721,backend,false,dtype).data();
        check(std::all_of(constant.begin(),constant.end(),[](float v){return v==std::numeric_limits<float>::infinity();}),"Native normal rejected or changed an overflowing mean");
        check(Tensor::normal({0},std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity(),5721,backend,false,dtype).raw_data().empty(),"Native empty normal nonfinite parameters differ");
        const double epsilon=dtype==DType::Float16?std::ldexp(1.,-10):dtype==DType::BFloat16?std::ldexp(1.,-7):std::ldexp(1.,-23);
        std::vector<float> unit=Tensor::uniform({65536},0,1,5721,backend,false,dtype).data();
        if(backend==Backend::CPU){
            const float divisor=dtype==DType::Float16?2048.f:dtype==DType::BFloat16?256.f:16777216.f;
            for(float value:unit)check(value>=0&&value<1&&std::floor(value*divisor)==value*divisor,"Native CPU uniform grid differs");
        }else{
            std::vector<float> narrow=Tensor::uniform({65536},1,1+epsilon,5721,backend,false,dtype).data();
            check(std::all_of(narrow.begin(),narrow.end(),[](float v){return v==1.f;}),"Native CUDA uniform upper endpoint was not reversed after storage rounding");
        }
        for(bool asynchronous:{false,true}){
            AllocationPool pool(backend);ExecutionStream producer(backend),consumer(backend);Tensor x;
            {StreamScope scope(producer,asynchronous,true);x=Tensor::uniform({513},0,1,5721,backend,true,dtype);}
            {StreamScope scope(consumer,asynchronous,true);Tensor y=add(x,Tensor::zeros(x.shape(),backend,false,dtype));
                check(y.raw_data()==x.raw_data(),"Floating random cross-stream dependency differs");
                y.backward(Tensor::ones(y.shape(),backend,false,dtype));
                check(Tensor(x.node()->gradient).raw_data()==Tensor::ones(x.shape(),backend,false,dtype).raw_data(),"Floating random cross-stream leaf derivative differs");}
            producer.synchronize();consumer.synchronize();
        }
    }
}
void floating_dropouts(Backend backend) {
    const double probability = .857142857157409;
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16, DType::Float8E4M3FN, DType::Float8E4M3FNUZ,
                       DType::Float8E5M2, DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor input = Tensor::from_data({7}, {1.f,-1.f,0.f,-0.f,std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}, backend, true, dtype);
        Tensor output = dropout(input, probability, 5721);
        std::vector<int> wide(10); wide[0] = 5721; wide[2] = 0x3f5b6db7; wide[3] = 1; wide[8] = 1;
        std::memcpy(wide.data() + 4, &probability, sizeof(probability));
        Tensor raw = dispatch_tensor("dropout", input.shape(), {input.detach()}, wide, dtype);
        check(raw.raw_data() == output.raw_data(), "Public dropout lost Double probability bits");
        Tensor inplace = Tensor::from_raw(input.shape(), dtype, input.raw_data(), backend);
        dispatch_tensor_into("dropout", inplace, {inplace}, wide);
        check(inplace.raw_data() == output.raw_data(), "Floating dropout in-place execution differs");
        check(dropout(input, .5, 17, false).node()->value == input.node()->value, "Evaluation dropout did not retain identity storage");
        Tensor empty = Tensor::ones({2,0,3}, backend, true, dtype);
        check(dropout(empty, .5, 17).node()->value == empty.node()->value, "Empty dropout did not retain identity storage");
        Tensor target = Tensor::zeros(input.shape(), backend, false, dtype);
        KernelParams valid = pointwise_descriptor(target, {input.detach()});
        valid.v[KernelParams::kAttributeSlot + 2] = 0x3f000000; valid.v[KernelParams::kAttributeSlot + 3] = 1;
        LocalFunction execute{[&](const KernelParams& p) { backend_dispatch("dropout", target.node()->value, {input.node()->value}, p); }};
        execute(valid);const std::vector<uint8_t> unchanged = target.raw_data();
        for (std::pair<int, int> mutation : std::vector<std::pair<int,int>>{{KernelParams::kOutputCountSlot,6},{KernelParams::kOutputRankSlot,9},
             {KernelParams::kOutputShapeSlot,6},{KernelParams::kOutputDTypeSlot,int(DType::Int32)},
             {KernelParams::input_slot(KernelInput::A),2},{KernelParams::input_dtype_slot(KernelInput::A),int(DType::Int32)},
             {KernelParams::input_slot(KernelInput::A)+KernelParams::kInputStrideOffset,2},
             {KernelParams::kAttributeSlot+2,int(0x7fc00000u)},{KernelParams::kAttributeSlot+3,2},
             {KernelParams::kAttributeSlot+8,2},{KernelParams::kAttributeSlot+9,2}}) {
            KernelParams bad=valid;bad.v[mutation.first]=mutation.second;rejects([&]{execute(bad);});
        }
        rejects([&]{backend_dispatch("dropout",target.node()->value,{},valid);});
        rejects([&]{backend_dispatch("dropout",target.node()->value,{input.node()->value,input.node()->value},valid);});
        check(target.raw_data()==unchanged,"Rejected floating dropout changed output");
        Tensor storage=Tensor::ones({10},backend,false,dtype);
        Tensor source=contiguous_view(storage,{8}),shifted=contiguous_view(storage,{8},2);
        rejects([&]{dispatch_tensor_into("dropout",shifted,{source},{17,0,0x3f000000,1});});
        Tensor saved=dropout(input,.5,17);input.assign(std::vector<float>(7,1));
        rejects([&]{saved.backward(Tensor::ones(saved.shape(),backend,false,dtype));});
    }
    const double tiny_probability=1.-1./(1.+std::ldexp(1.,-24));
    Tensor leaf=Tensor::ones({1},backend,true);
    Tensor result=dropout(leaf,tiny_probability,5721);
    result.backward(Tensor::ones({1},backend));
    check(result.data()==std::vector<float>{std::nextafter(1.f,2.f)},"Dropout forward reciprocal rounding differs");
    check(leaf.grad()==std::vector<float>{backend==Backend::CUDA?1.f:std::nextafter(1.f,2.f)},"Dropout backward reciprocal rounding differs");
    for (bool asynchronous : {false,true}) {
        AllocationPool pool(backend);ExecutionStream producer(backend),consumer(backend);Tensor x;
        {StreamScope scope(producer,asynchronous,true);x=Tensor::ones({513},backend,true,DType::Float16);}
        {
            StreamScope scope(consumer,asynchronous,true);
            Tensor y=dropout(x,.1234567891234,5721);y.backward(Tensor::ones(x.shape(),backend,false,x.dtype()));
            check(y.raw_data()==Tensor(x.node()->gradient).raw_data(),"Floating dropout cross-stream mask differs");
            check(y.raw_data()==dropout(x.detach(),.1234567891234,5721).raw_data(),"Floating dropout stream replay differs");
        }
        producer.synchronize();consumer.synchronize();
    }
}
void randoms(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor first = Tensor::uniform({4097}, -2, 2, 42, backend, true, dtype);
        check(first.raw_data() == Tensor::uniform({4097}, -2, 2, 42, backend, false, dtype).raw_data(), "Low precision uniform replay differs");
        check(first.raw_data() != Tensor::uniform({4097}, -2, 2, 43, backend, false, dtype).raw_data(), "Uniform seed isolation differs");
        std::vector<float> values = first.data();
        check(std::all_of(values.begin(), values.end(), [](float x) { return x >= -2 && x <= 2; }), "Uniform support differs");
        first.backward(Tensor::ones(first.shape(), backend, false, dtype));
        check(Tensor(first.node()->gradient).raw_data() == Tensor::ones(first.shape(), backend, false, dtype).raw_data(), "Random tensor is not a differentiable leaf");
        Tensor normal = Tensor::normal({4097}, .25, 1.5, 41, backend, false, dtype);
        check(normal.raw_data() == Tensor::normal({4097}, .25, 1.5, 41, backend, false, dtype).raw_data(), "Low precision normal replay differs");
        Tensor output = Tensor::zeros({4097}, backend, false, dtype);
        KernelParams valid = descriptor(output, output); valid.v[KernelParams::input_slot(KernelInput::A)] = -1;
        valid.v[KernelParams::kAttributeSlot] = 41;
        valid.v[KernelParams::kAttributeSlot + 2] = 0x3e800000; valid.v[KernelParams::kAttributeSlot + 3] = 0x3fc00000;
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("normal", output.node()->value, {}, p); }};
        raw(valid); check(output.raw_data() == normal.raw_data(), "Legacy random parameter encoding differs");
        KernelParams offset = valid; offset.v[KernelParams::kAttributeSlot + 1] = 7; raw(offset);
        check(output.raw_data() != normal.raw_data(), "Random stream offsets collided");
        const std::vector<uint8_t> unchanged = output.raw_data();
        for (std::pair<int, int> mutation : std::vector<std::pair<int, int>>{{KernelParams::kOutputCountSlot, 4096}, {KernelParams::kOutputRankSlot, 9},
             {KernelParams::kOutputShapeSlot, 4096}, {KernelParams::kOutputDTypeSlot, int(DType::Float32)},
             {KernelParams::input_slot(KernelInput::A), 0}, {KernelParams::kAttributeSlot + 8, 2}, {KernelParams::kAttributeSlot + 3, int(0xbf800000u)}}) {
            KernelParams bad = valid; bad.v[mutation.first] = mutation.second; rejects([&] { raw(bad); });
        }
        rejects([&] { backend_dispatch("normal", output.node()->value, {first.node()->value}, valid); });
        check(output.raw_data() == unchanged, "Rejected random descriptor changed output");
        Tensor x = Tensor::ones({4097}, backend, true, dtype);
        Tensor dropped = dropout(x, .5, 17);
        dropped.backward(Tensor::ones(x.shape(), backend, false, dtype));
        check(dropped.raw_data() == Tensor(x.node()->gradient).raw_data(), "Dropout replayed a different backward mask");
        Tensor edge_leaf = Tensor::ones({4}, backend, true, dtype);
        Tensor edge_gradient = Tensor::from_data({4}, {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), -0.f}, backend, false, dtype);
        dropout(edge_leaf, 1., 17).backward(edge_gradient);
        std::vector<float> edge_values = edge_gradient.data(); for (float& value : edge_values) value *= 0.f;
        compare(Tensor(edge_leaf.node()->gradient), specification(Tensor::from_data({4}, edge_values, backend, false, dtype)), "Dropout nonfinite upstream control");
        check(dropout(x, 0, 17).raw_data() == x.raw_data() && dropout(x, .5, 17, false).raw_data() == x.raw_data(), "Dropout identity differs");
        Tensor inplace = Tensor::ones({4097}, backend, false, dtype);
        dispatch_tensor_into("dropout", inplace, {inplace}, {17, 0, 0x3f000000, 1});
        check(inplace.raw_data() == dropped.raw_data(), "Packed in-place dropout differs");
        KernelParams dropout_valid = pointwise_descriptor(output, {x.detach()});
        dropout_valid.v[KernelParams::kAttributeSlot + 2] = 0x3f000000; dropout_valid.v[KernelParams::kAttributeSlot + 3] = 1;
        LocalFunction dropout_raw{[&](const KernelParams& p) { backend_dispatch("dropout", output.node()->value, {x.node()->value}, p); }};
        for (std::pair<int, int> mutation : std::vector<std::pair<int, int>>{{KernelParams::input_slot(KernelInput::A), 2},
             {KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset, 2},
             {KernelParams::input_dtype_slot(KernelInput::A), int(DType::Float32)}, {KernelParams::kAttributeSlot + 3, 2},
             {KernelParams::kAttributeSlot + 2, int(0xbf800000u)}, {KernelParams::kAttributeSlot + 2, int(0x7fc00000u)}}) {
            KernelParams bad = dropout_valid; bad.v[mutation.first] = mutation.second; rejects([&] { dropout_raw(bad); });
        }
        rejects([&] { backend_dispatch("dropout", output.node()->value, {}, dropout_valid); });
        rejects([&] { backend_dispatch("dropout", output.node()->value, {x.node()->value, x.node()->value}, dropout_valid); });
        Tensor storage = Tensor::ones({10}, backend, false, dtype);
        Tensor source = contiguous_view(storage, {8}), shifted = contiguous_view(storage, {8}, 2);
        rejects([&] { dispatch_tensor_into("dropout", shifted, {source}, {17, 0, 0x3f000000, 1}); });
        Tensor constant = Tensor::normal({7}, 1./3, 0., 17, backend, false, dtype);
        check(constant.raw_data() == Tensor::from_data({7}, std::vector<float>(7, 1.f/3), backend, false, dtype).raw_data(), "Zero-variance normal differs");
        for (std::string op : {std::string("uniform"), std::string("normal")}) {
            Tensor empty = op == "uniform" ? Tensor::uniform({2, 0, 3}, 0, 1, 17, backend, false, dtype) : Tensor::normal({2, 0, 3}, 0, 1, 17, backend, false, dtype);
            check(empty.shape() == Shape({2, 0, 3}) && empty.raw_data().empty(), "Empty random tensor differs");
        }
        rejects([&] { Tensor::uniform({7}, -double(std::numeric_limits<float>::max()), double(std::numeric_limits<float>::max()), 17, backend, false, dtype); });
        rejects([&] { Tensor::uniform({7}, 2., 1., 17, backend, false, dtype); });
        rejects([&] { Tensor::normal({7}, 0., -1., 17, backend, false, dtype); });
        rejects([&] { dropout(x, std::numeric_limits<double>::quiet_NaN(), 17, false); });
        Tensor saved = dropout(x, .5, 17); x.assign(std::vector<float>(4097, 2));
        rejects([&] { saved.backward(Tensor::ones(saved.shape(), backend, false, dtype)); });
        if (dtype == DType::Float4E2M1FN) {
            check((first.raw_data().back() & 0xf0) == 0 && (normal.raw_data().back() & 0xf0) == 0 && (dropped.raw_data().back() & 0xf0) == 0, "Random packed padding is dirty");
            Tensor dirty = Tensor::from_raw({3}, dtype, {0x21, 0xfa}, backend);
            Tensor clean = dispatch_tensor("dropout", {3}, {dirty}, {17, 0, 0, 0}, dtype);
            check(clean.raw_data() == std::vector<uint8_t>({0x21, 0x0a}), "Dropout identity did not clear unused packed padding");
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor x;
        { StreamScope scope(producer, asynchronous, true); x = Tensor::normal({4097}, 0, 1, 31, backend, true, DType::Float4E2M1FN); }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor result = dropout(x, .5, 17);
            result.backward(Tensor::ones(result.shape(), backend, false, result.dtype()));
            Tensor expected = dropout(Tensor::ones(x.shape(), backend, false, x.dtype()), .5, 17);
            check(Tensor(x.node()->gradient).raw_data() == expected.raw_data(), "Random cross-stream backward mask differs");
            check(result.raw_data() == dropout(x.detach(), .5, 17).raw_data(), "Random cross-stream forward replay differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void attention(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor scores = Tensor::ones({3, 3}, backend, true, dtype), mask = Tensor::zeros({}, backend, true, dtype);
        Tensor probability = attention_softmax(scores, mask, 2);
        check(probability.raw_data() == Tensor::from_data({3, 3}, std::vector<float>(9, 1.f / 3), backend, false, dtype).raw_data(), "Low precision attention softmax differs");
        probability.backward(Tensor::ones({3, 3}, backend, false, dtype));
        check(Tensor(scores.node()->gradient).raw_data() == Tensor::zeros({3, 3}, backend, false, dtype).raw_data(), "Attention score derivative differs");
        check(Tensor(mask.node()->gradient).raw_data() == Tensor::zeros({}, backend, false, dtype).raw_data(), "Attention mask derivative differs");
        Tensor q = Tensor::ones({1, 4, 3, 2}, backend, true, dtype), k = Tensor::ones({1, 2, 3, 2}, backend, true, dtype);
        Tensor v = Tensor::from_data({1, 2, 3, 1}, std::vector<float>(6, 2), backend, true, dtype);
        Tensor m = Tensor::zeros({}, backend, true, dtype);
        Tensor result = tiled_attention(q, k, v, m, 2, 2);
        check(result.raw_data() == Tensor::from_data({1, 4, 3, 1}, std::vector<float>(12, 2), backend, false, dtype).raw_data(), "Low precision grouped attention differs");
        result.backward(Tensor::ones(result.shape(), backend, false, dtype));
        check(Tensor(q.node()->gradient).raw_data() == Tensor::zeros(q.shape(), backend, false, dtype).raw_data(), "Grouped query derivative differs");
        check(Tensor(k.node()->gradient).raw_data() == Tensor::zeros(k.shape(), backend, false, dtype).raw_data(), "Grouped key derivative differs");
        check(Tensor(v.node()->gradient).raw_data() == Tensor::from_data(v.shape(), std::vector<float>(6, 2), backend, false, dtype).raw_data(), "Grouped value derivative differs");
        Tensor output = Tensor::zeros({3, 3}, backend, false, dtype);
        KernelParams valid = pointwise_descriptor(output, {scores.detach(), mask.detach()});
        valid.v[KernelParams::kAttributeSlot] = 1; valid.v[KernelParams::kAttributeSlot + 1] = 0x40000000;
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("attention_softmax", output.node()->value, {scores.node()->value, mask.node()->value}, p); }};
        raw(valid); const std::vector<uint8_t> unchanged = output.raw_data();
        for (std::pair<int, int> mutation : std::vector<std::pair<int, int>>{{KernelParams::kOutputCountSlot, 8}, {KernelParams::kOutputRankSlot, 9},
             {KernelParams::kOutputShapeSlot, 2}, {KernelParams::kOutputDTypeSlot, int(DType::Float32)},
             {KernelParams::input_slot(KernelInput::A), 9}, {KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset, 1},
             {KernelParams::input_dtype_slot(KernelInput::B), int(DType::Float32)}, {KernelParams::kAttributeSlot, 2},
             {KernelParams::kAttributeSlot + 2, 1}, {KernelParams::kAttributeSlot + 7, 2}, {KernelParams::kAttributeSlot + 8, 1}}) {
            KernelParams bad = valid; bad.v[mutation.first] = mutation.second; rejects([&] { raw(bad); });
        }
        rejects([&] { backend_dispatch("attention_softmax", output.node()->value, {scores.node()->value}, valid); });
        rejects([&] { backend_dispatch("attention_softmax", output.node()->value, {scores.node()->value, mask.node()->value, mask.node()->value}, valid); });
        rejects([&] { attention_softmax(scores, Tensor::zeros({}, backend)); });
        rejects([&] { attention_softmax(scores, Tensor::zeros({4}, backend, false, dtype)); });
        check(output.raw_data() == unchanged, "Rejected attention descriptor wrote output bytes");
        Tensor inplace = scores.detach(); dispatch_tensor_into("attention_softmax", inplace, {inplace, mask.detach()}, {1, 0x40000000});
        check(inplace.raw_data() == probability.raw_data(), "In-place attention softmax differs");
        Tensor alias_mask = Tensor::zeros({3, 3}, backend, false, dtype);
        rejects([&] { dispatch_tensor_into("attention_softmax", alias_mask, {output, alias_mask}, {1, 0x40000000}); });
        Tensor tiled_output = Tensor::zeros(q.shape(), backend, false, dtype);
        Tensor vv = Tensor::ones(k.shape(), backend, false, dtype);
        KernelParams tiled_valid = pointwise_descriptor(tiled_output, {q.detach(), k.detach(), vv, m.detach()});
        tiled_valid.v[KernelParams::kAttributeSlot] = 0x40000000; tiled_valid.v[KernelParams::kAttributeSlot + 1] = 2;
        LocalFunction tiled_raw{[&](const KernelParams& p) { backend_dispatch("tiled_attention", tiled_output.node()->value, {q.node()->value, k.node()->value, vv.node()->value, m.node()->value}, p); }};
        tiled_raw(tiled_valid); const std::vector<uint8_t> tiled_unchanged = tiled_output.raw_data();
        for (std::pair<int, int> mutation : std::vector<std::pair<int, int>>{{KernelParams::kOutputCountSlot, 23},
             {KernelParams::input_slot(KernelInput::B), 3}, {KernelParams::input_slot(KernelInput::D), 9},
             {KernelParams::input_slot(KernelInput::B) + KernelParams::kInputShapeOffset + 1, 3},
             {KernelParams::kAttributeSlot + 1, 0}, {KernelParams::kAttributeSlot + 2, 2}, {KernelParams::kAttributeSlot + 4, 2},
             {KernelParams::kAttributeSlot + 7, -1}, {KernelParams::kAttributeSlot + 8, 1}}) {
            KernelParams bad = tiled_valid; bad.v[mutation.first] = mutation.second; rejects([&] { tiled_raw(bad); });
        }
        rejects([&] { backend_dispatch("tiled_attention", tiled_output.node()->value, {q.node()->value, k.node()->value, vv.node()->value}, tiled_valid); });
        rejects([&] { dispatch_tensor_into("tiled_attention", q.detach(), {q.detach(), k.detach(), vv, m.detach()}, {0x40000000, 2}); });
        rejects([&] { tiled_attention(q, k, vv, m, 2, 2, false, 0, AttentionTiling(2)); });
        rejects([&] { tiled_attention(q, k, vv, Tensor::ones({}, backend)); });
        check(tiled_output.raw_data() == tiled_unchanged, "Rejected tiled descriptor wrote output bytes");
        if (dtype == DType::Float4E2M1FN) {
            check((probability.raw_data().back() & 0xf0) == 0, "Packed attention padding is dirty");
            check((Tensor(scores.node()->gradient).raw_data().back() & 0xf0) == 0, "Packed attention derivative padding is dirty");
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor q, k, v, m;
        {
            StreamScope scope(producer, asynchronous, true);
            q = Tensor::ones({1, 2, 3, 2}, backend, false, DType::Float4E2M1FN);
            k = Tensor::ones({1, 1, 3, 2}, backend, false, q.dtype());
            v = Tensor::ones({1, 1, 3, 1}, backend, true, q.dtype());
            m = Tensor::zeros({}, backend, true, q.dtype());
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor result = tiled_attention(q, k, v, m, 2, 2);
            result.backward(Tensor::ones(result.shape(), backend, false, result.dtype()));
            check(result.data() == std::vector<float>(6, 1), "Attention stream dependency differs");
            check(Tensor(v.node()->gradient).data() == std::vector<float>(3, 2), "Attention gradient stream dependency differs");
            Tensor saved = tiled_attention(q, k, v, m, 2, 2); m.assign({1});
            rejects([&] { saved.backward(Tensor::ones(saved.shape(), backend, false, saved.dtype())); });
        }
        producer.synchronize(); consumer.synchronize();
    }
    for (bool reuse : {false, true}) {
        const size_t before = memory_stats(backend).live_bytes();
        {
            AllocationPool pool(backend, 0, reuse);
            Tensor q = Tensor::ones({1, 4, 257, 8}, backend, false, DType::Float4E2M1FN);
            Tensor k = Tensor::ones({1, 2, 512, 8}, backend, false, q.dtype());
            Tensor v = Tensor::from_data(k.shape(), std::vector<float>(8192, 2), backend, false, q.dtype());
            Tensor m = Tensor::zeros({}, backend, false, q.dtype());
            tiled_attention(slice(q, 2, 0, 1), k, v, m, 2, 8).data(); pool.clear();
            const size_t budget = memory_stats(backend).live_bytes() + 1024 * 1024;
            MemoryLimit limit(backend, budget);
            Tensor output = tiled_attention(q, k, v, m, 2, 8);
            check(output.data() == std::vector<float>(8224, 2), "Bounded low precision attention differs");
            check(limit.peak_bytes() <= budget, "Low precision attention exceeded tile workspace bound");
            bool rejected = false;
            try { matmul_transposed(cast(q, DType::Float32), Tensor::ones({1, 4, 512, 8}, backend)).data(); }
            catch (const MemoryLimitExceeded&) { rejected = true; }
            check(rejected, "Dense scores unexpectedly fit the low precision tile workspace bound");
        }
        check(memory_stats(backend).live_bytes() == before, "Low precision attention leaked workspace storage");
    }
    Tensor x = Tensor::ones({2, 3}, backend), y = Tensor::ones({2, 3}, backend);
    KernelParams intermediate = pointwise_descriptor(y, {x});
    intermediate.v[KernelParams::kAttributeSlot] = 1; intermediate.v[KernelParams::kAttributeSlot + 8] = 2;
    LocalFunction probabilities_raw{[&](const KernelParams& p) { backend_dispatch("attention_softmax", y.node()->value, {x.node()->value}, p); }};
    probabilities_raw(intermediate);
    for (int slot : {1, 2, 5, 6, 7}) {
        KernelParams bad = intermediate; bad.v[KernelParams::kAttributeSlot + slot] = 1;
        rejects([&] { probabilities_raw(bad); });
    }
    KernelParams alias = pointwise_descriptor(x, {x}); alias.v[KernelParams::kAttributeSlot] = 1; alias.v[KernelParams::kAttributeSlot + 8] = 2;
    rejects([&] { backend_dispatch("attention_softmax", x.node()->value, {x.node()->value}, alias); });
    Tensor reduced = Tensor::zeros({1, 3}, backend);
    KernelParams reduction = pointwise_descriptor(reduced, {x}); reduction.v[KernelParams::kAttributeSlot] = 1; reduction.v[KernelParams::kAttributeSlot + 8] = 5;
    backend_dispatch("attention_softmax", reduced.node()->value, {x.node()->value}, reduction);
    check(reduced.data() == std::vector<float>(3, 2), "Attention intermediate reduction differs");
    for (int value : {0, 2, 4, -1}) {
        KernelParams bad = reduction; bad.v[KernelParams::kAttributeSlot] = value;
        rejects([&] { backend_dispatch("attention_softmax", reduced.node()->value, {x.node()->value}, bad); });
    }
    Tensor a = Tensor::ones({1, 2, 3, 4}, backend), b = Tensor::ones({1, 2, 3, 5}, backend);
    Tensor product = Tensor::zeros({1, 2, 4, 5}, backend);
    KernelParams matrix = pointwise_descriptor(product, {a, b}); matrix.v[KernelParams::kAttributeSlot] = 1; matrix.v[KernelParams::kAttributeSlot + 8] = 4;
    backend_dispatch("attention_softmax", product.node()->value, {a.node()->value, b.node()->value}, matrix);
    check(product.data() == std::vector<float>(40, 3), "Attention transposed matrix intermediate differs");
    for (int slot : {0, 1, 2, 3, 7}) {
        KernelParams bad = matrix; bad.v[KernelParams::kAttributeSlot + slot] = 2;
        rejects([&] { backend_dispatch("attention_softmax", product.node()->value, {a.node()->value, b.node()->value}, bad); });
    }
}
void fusions(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor gate = Tensor::ones({3}, backend, true, dtype);
        Tensor up = Tensor::from_data({}, {2}, backend, true, dtype);
        Tensor result = silu_mul(gate, up);
        const float sigmoid = 1.f / (1.f + std::exp(-1.f));
        check(result.raw_data() == Tensor::from_data({3}, std::vector<float>(3, 2 * sigmoid), backend, false, dtype).raw_data(), "Low precision SiLU multiplication differs");
        result.backward(Tensor::ones({3}, backend, false, dtype));
        check(Tensor(gate.node()->gradient).raw_data() == Tensor::from_data({3}, std::vector<float>(3, 2 * sigmoid * (2 - sigmoid)), backend, false, dtype).raw_data(), "Fused gate derivative differs");
        check(Tensor(up.node()->gradient).raw_data() == Tensor::from_data({}, {3 * sigmoid}, backend, false, dtype).raw_data(), "Fused broadcast derivative rounded before reduction");
        Tensor x = Tensor::from_data({1, 4}, {1, 2, 1, 2}, backend, true, dtype);
        Tensor cosine = Tensor::ones({4}, backend, true, dtype), sine = Tensor::ones({4}, backend, true, dtype);
        Tensor rotated = rotary_embedding(x, cosine, sine);
        check(rotated.raw_data() == Tensor::from_data({1, 4}, {0, 0, 2, 4}, backend, false, dtype).raw_data(), "Low precision rotary embedding differs");
        rotated.backward(Tensor::ones({1, 4}, backend, false, dtype));
        check(Tensor(x.node()->gradient).raw_data() == Tensor::from_data({1, 4}, {2, 2, 0, 0}, backend, false, dtype).raw_data(), "Rotary input derivative differs");
        check(Tensor(cosine.node()->gradient).raw_data() == Tensor::from_data({4}, {1, 2, 1, 2}, backend, false, dtype).raw_data(), "Rotary cosine derivative differs");
        check(Tensor(sine.node()->gradient).raw_data() == Tensor::from_data({4}, {-1, -2, 1, 2}, backend, false, dtype).raw_data(), "Rotary sine derivative differs");
        Tensor input = Tensor::ones({3}, backend, false, dtype), factor = Tensor::ones({3}, backend, false, dtype);
        Tensor output = Tensor::zeros({3}, backend, false, dtype);
        KernelParams valid = pointwise_descriptor(output, {input, factor});
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("silu_mul", output.node()->value, {input.node()->value, factor.node()->value}, p); }};
        raw(valid); const std::vector<uint8_t> unchanged = output.raw_data();
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputShapeSlot] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B) + KernelParams::kInputShapeOffset] = 4; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::C)] = 0; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("silu_mul", output.node()->value, {input.node()->value}, valid); });
        rejects([&] { backend_dispatch("silu_mul", output.node()->value, {input.node()->value, factor.node()->value, factor.node()->value}, valid); });
        rejects([&] { silu_mul(input, Tensor::ones({3}, backend)); });
        rejects([&] { silu_mul(input, Tensor::ones({4}, backend, false, dtype)); });
        check(output.raw_data() == unchanged, "Rejected fused descriptor wrote output bytes");
        Tensor inplace = Tensor::ones({3}, backend, false, dtype);
        dispatch_tensor_into("silu_mul", inplace, {inplace, factor});
        check(inplace.raw_data() == unchanged, "In-place fused multiplication differs");
        Tensor odd = rotary_embedding(Tensor::ones({3, 2}, backend, false, dtype), Tensor::ones({2}, backend, false, dtype), Tensor::ones({}, backend, false, dtype));
        check(odd.shape() == Shape({3, 2}), "Rotary broadcast shape differs");
        Tensor rotary_input = Tensor::ones({2, 4}, backend, false, dtype);
        Tensor rotary_factor = Tensor::ones({4}, backend, false, dtype);
        KernelParams rotary_valid = pointwise_descriptor(rotary_input, {rotary_input, rotary_factor, rotary_factor});
        rejects([&] { backend_dispatch("rotary_embedding", rotary_input.node()->value, {rotary_input.node()->value, rotary_factor.node()->value, rotary_factor.node()->value}, rotary_valid); });
        rejects([&] { rotary_embedding(input, factor, factor); });
        rejects([&] { rotary_embedding(Tensor::ones({}, backend, false, dtype), Tensor::ones({}, backend, false, dtype), Tensor::ones({}, backend, false, dtype)); });
        rejects([&] { rotary_embedding(Tensor::ones({1, 4}, backend, false, dtype), Tensor::ones({2, 4}, backend, false, dtype), rotary_factor); });
        rejects([&] { rotary_embedding(rotary_input, Tensor::ones({4}, backend), rotary_factor); });
        if (dtype == DType::Float4E2M1FN) {
            check((inplace.raw_data().back() & 0xf0) == 0, "Packed fused output padding is dirty");
            check((Tensor(gate.node()->gradient).raw_data().back() & 0xf0) == 0, "Packed fused derivative padding is dirty");
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor x, c, s;
        {
            StreamScope scope(producer, asynchronous, true);
            x = Tensor::ones({3, 2}, backend, true, DType::Float4E2M1FN);
            c = Tensor::ones({2}, backend, false, x.dtype()); s = Tensor::ones({2}, backend, false, x.dtype());
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor result = rotary_embedding(x, c, s);
            result.backward(Tensor::ones(result.shape(), backend, false, result.dtype()));
            check(result.data() == std::vector<float>({0, 2, 0, 2, 0, 2}), "Fused stream dependency differs");
            check(Tensor(x.node()->gradient).data() == std::vector<float>({2, 0, 2, 0, 2, 0}), "Fused gradient stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void losses(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor x = Tensor::ones({3}, backend, true, dtype), y = Tensor::zeros({3}, backend, true, dtype);
        Tensor mse = mse_loss(x, y);
        check(mse.raw_data() == Tensor::ones({}, backend, false, dtype).raw_data(), "Low precision MSE differs");
        mse.backward(Tensor::ones({}, backend, false, dtype));
        check(Tensor(x.node()->gradient).raw_data() == Tensor::from_data({3}, std::vector<float>(3, 2.f / 3), backend, false, dtype).raw_data(), "Low precision MSE input derivative differs");
        check(Tensor(y.node()->gradient).raw_data() == Tensor::from_data({3}, std::vector<float>(3, -2.f / 3), backend, false, dtype).raw_data(), "Low precision MSE target derivative differs");
        Tensor scalar = Tensor::ones({}, backend, true, dtype);
        Tensor broadcasted = mse_loss(scalar, Tensor::zeros({2, 3}, backend, false, dtype));
        broadcasted.backward(Tensor::ones({}, backend, false, dtype));
        check(Tensor(scalar.node()->gradient).raw_data() == Tensor::from_data({}, {2}, backend, false, dtype).raw_data(), "Broadcast MSE derivative rounded before reduction");
        Tensor logits = Tensor::ones({2, 3}, backend, true, dtype);
        Tensor ids = Tensor::from_int_data({2}, {0, 2}, backend);
        Tensor ce = cross_entropy(logits, ids);
        check(ce.raw_data() == Tensor::from_data({}, {std::log(3.f)}, backend, false, dtype).raw_data(), "Low precision cross entropy differs");
        ce.backward(Tensor::from_data({}, {2}, backend, false, dtype));
        check(Tensor(logits.node()->gradient).raw_data() == Tensor::from_data({2, 3}, {-2.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, 1.f / 3, -2.f / 3}, backend, false, dtype).raw_data(), "Low precision cross entropy derivative differs");
        Tensor output = Tensor::zeros({}, backend, false, dtype);
        KernelParams valid = pointwise_descriptor(output, {logits, ids});
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("cross_entropy", output.node()->value, {logits.node()->value, ids.node()->value}, p); }};
        raw(valid); const std::vector<uint8_t> unchanged = output.raw_data();
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B)] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::C)] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = 1; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("cross_entropy", output.node()->value, {logits.node()->value}, valid); });
        rejects([&] { backend_dispatch("cross_entropy", output.node()->value, {logits.node()->value, ids.node()->value, logits.node()->value}, valid); });
        for (int value : {-1, 3}) {
            Tensor invalid_ids = Tensor::from_int_data({2}, {0, value}, backend);
            KernelParams invalid = pointwise_descriptor(output, {logits, invalid_ids});
            rejects([&] { backend_dispatch("cross_entropy", output.node()->value, {logits.node()->value, invalid_ids.node()->value}, invalid); });
            rejects([&] { cross_entropy(logits, invalid_ids); });
        }
        check(output.raw_data() == unchanged, "Rejected loss descriptor wrote output bytes");
        if (dtype != DType::Float4E2M1FN) {
            Tensor loss_alias = contiguous_view(logits, {}, 0);
            KernelParams alias = pointwise_descriptor(loss_alias, {logits, ids});
            rejects([&] { backend_dispatch("cross_entropy", loss_alias.node()->value, {logits.node()->value, ids.node()->value}, alias); });
        }
        KernelParams mse_valid = pointwise_descriptor(output, {x, y});
        bad = mse_valid; bad.v[KernelParams::kAttributeSlot] = 2;
        rejects([&] { backend_dispatch("mse_loss", output.node()->value, {x.node()->value, y.node()->value}, bad); });
        rejects([&] { mse_loss(x, Tensor::ones({3}, backend)); });
        rejects([&] { mse_loss(x, Tensor::ones({4}, backend, false, dtype)); });
        Tensor seed = Tensor::ones({}, backend, false, dtype);
        Tensor gradient = Tensor::zeros({3}, backend, false, dtype);
        KernelParams derivative = pointwise_descriptor(gradient, {x, y, seed}); derivative.v[KernelParams::kAttributeSlot] = 1;
        backend_dispatch("mse_loss", gradient.node()->value, {x.node()->value, y.node()->value, seed.node()->value}, derivative);
        check(gradient.raw_data() == Tensor(x.node()->gradient).raw_data(), "Raw MSE derivative differs");
        Tensor upstream_vector = Tensor::ones({1}, backend, false, dtype);
        KernelParams invalid_derivative = pointwise_descriptor(gradient, {x, y, upstream_vector}); invalid_derivative.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("mse_loss", gradient.node()->value, {x.node()->value, y.node()->value, upstream_vector.node()->value}, invalid_derivative); });
        KernelParams alias_derivative = pointwise_descriptor(x, {x, y, seed}); alias_derivative.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("mse_loss", x.node()->value, {x.node()->value, y.node()->value, seed.node()->value}, alias_derivative); });
        if (dtype == DType::Float4E2M1FN) check((gradient.raw_data().back() & 0xf0) == 0, "Packed loss derivative padding is dirty");
        Tensor odd_logits = Tensor::ones({1, 3}, backend, true, dtype);
        if (dtype == DType::Float4E2M1FN) {
            Tensor tail = contiguous_view(odd_logits, {}, 2);
            Tensor one_id = Tensor::from_int_data({1}, {1}, backend);
            KernelParams alias = pointwise_descriptor(tail, {odd_logits, one_id});
            rejects([&] { backend_dispatch("cross_entropy", tail.node()->value, {odd_logits.node()->value, one_id.node()->value}, alias); });
        }
        Tensor odd_loss = cross_entropy(odd_logits, Tensor::from_int_data({1}, {1}, backend));
        odd_loss.backward(Tensor::ones({}, backend, false, dtype));
        if (dtype == DType::Float4E2M1FN) check((Tensor(odd_logits.node()->gradient).raw_data().back() & 0xf0) == 0, "Packed cross entropy derivative padding is dirty");
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor input, ids;
        {
            StreamScope scope(producer, asynchronous, true);
            input = Tensor::ones({1, 3}, backend, true, DType::Float4E2M1FN);
            ids = Tensor::from_int_data({1}, {1}, backend);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor loss = cross_entropy(input, ids);
            loss.backward(Tensor::ones({}, backend, false, loss.dtype()));
            check(loss.raw_data() == Tensor::from_data({}, {std::log(3.f)}, backend, false, loss.dtype()).raw_data(), "Loss stream dependency differs");
            check(bool(input.node()->gradient), "Loss gradient stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void spatials(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor x = Tensor::ones({1, 1, 3, 3}, backend, true, dtype);
        Tensor w = Tensor::ones({1, 1, 2, 2}, backend, true, dtype);
        Tensor b = Tensor::zeros({1}, backend, true, dtype);
        Tensor y = conv2d(x, w, {});
        check(y.raw_data() == Tensor::from_data({1, 1, 2, 2}, {4, 4, 4, 4}, backend, false, dtype).raw_data(), "Low precision convolution differs");
        y.backward(Tensor::ones(y.shape(), backend, false, dtype));
        check(Tensor(x.node()->gradient).raw_data() == Tensor::from_data(x.shape(), {1, 2, 1, 2, 4, 2, 1, 2, 1}, backend, false, dtype).raw_data(),
              "Low precision convolution input gradient differs");
        check(Tensor(w.node()->gradient).raw_data() == Tensor::from_data(w.shape(), {4, 4, 4, 4}, backend, false, dtype).raw_data(),
              "Low precision convolution weight gradient differs");
        Tensor biased = conv2d(x.detach(), w.detach(), b);
        biased.backward(Tensor::ones(biased.shape(), backend, false, dtype));
        check(Tensor(b.node()->gradient).raw_data() == Tensor::from_data({1}, {4}, backend, false, dtype).raw_data(), "Convolution bias gradient differs");
        for (bool maximum : {false, true}) {
            Tensor input = Tensor::from_data({1, 1, 2, 2}, {1, 1, 1, 4}, backend, true, dtype);
            Tensor output = maximum ? max_pool2d(input, 2) : avg_pool2d(input, 2);
            const std::vector<float> decoded = input.data();
            const float expected = maximum ? 4.f : (decoded[0] + decoded[1] + decoded[2] + decoded[3]) / 4.f;
            check(output.raw_data() == Tensor::from_data({1, 1, 1, 1}, {expected}, backend, false, dtype).raw_data(), "Low precision pooling differs");
            output.backward(Tensor::from_data({1, 1, 1, 1}, {4}, backend, false, dtype));
            check(Tensor(input.node()->gradient).raw_data() == Tensor::from_data(input.shape(), maximum ? std::vector<float>{0, 0, 0, 4} : std::vector<float>{1, 1, 1, 1}, backend, false, dtype).raw_data(), "Low precision pooling gradient differs");
        }
        Tensor output = Tensor::zeros({1, 1, 2, 2}, backend, false, dtype);
        KernelParams valid = pointwise_descriptor(output, {x, w});
        const int attributes[]{1, 1, 0, 0, 1, 1, 1};
        std::copy(std::begin(attributes), std::end(attributes), valid.v + KernelParams::kAttributeSlot);
        LocalFunction raw{[&](const KernelParams& descriptor) { backend_dispatch("conv2d", output.node()->value, {x.node()->value, w.node()->value}, descriptor); }};
        raw(valid); const std::vector<uint8_t> unchanged = output.raw_data();
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 3; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 3; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputShapeSlot + 1] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::C)] = 0; rejects([&] { raw(bad); });
        for (int slot : {0, 1, 4, 5, 6}) { bad = valid; bad.v[KernelParams::kAttributeSlot + slot] = 0; rejects([&] { raw(bad); }); }
        for (int slot : {2, 3}) { bad = valid; bad.v[KernelParams::kAttributeSlot + slot] = -1; rejects([&] { raw(bad); }); }
        bad = valid; bad.v[KernelParams::kAttributeSlot + 6] = 2; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("conv2d", output.node()->value, {x.node()->value}, valid); });
        rejects([&] { backend_dispatch("conv2d", output.node()->value, {x.node()->value, w.node()->value, x.node()->value}, valid); });
        Tensor mixed_weight = Tensor::ones(w.shape(), backend);
        KernelParams mixed = pointwise_descriptor(output, {x, mixed_weight});
        std::copy(std::begin(attributes), std::end(attributes), mixed.v + KernelParams::kAttributeSlot);
        rejects([&] { backend_dispatch("conv2d", output.node()->value, {x.node()->value, mixed_weight.node()->value}, mixed); });
        KernelParams alias = pointwise_descriptor(w, {x, w});
        std::copy(std::begin(attributes), std::end(attributes), alias.v + KernelParams::kAttributeSlot);
        rejects([&] { backend_dispatch("conv2d", w.node()->value, {x.node()->value, w.node()->value}, alias); });
        check(output.raw_data() == unchanged, "Rejected spatial descriptor wrote output bytes");
        Tensor small = Tensor::ones({1, 1, 2, 2}, backend, false, dtype);
        Tensor single = Tensor::zeros({1, 1, 1, 1}, backend, false, dtype);
        KernelParams pool = pointwise_descriptor(single, {small});
        const int pool_attributes[]{2, 2, 2, 2, 0, 0, 1};
        std::copy(std::begin(pool_attributes), std::end(pool_attributes), pool.v + KernelParams::kAttributeSlot);
        for (const std::string op : {"max_pool2d", "avg_pool2d"}) {
            backend_dispatch(op, single.node()->value, {small.node()->value}, pool);
            bad = pool; bad.v[KernelParams::kAttributeSlot] = 0;
            rejects([&] { backend_dispatch(op, single.node()->value, {small.node()->value}, bad); });
            bad = pool; bad.v[KernelParams::kAttributeSlot + 4] = 2;
            rejects([&] { backend_dispatch(op, single.node()->value, {small.node()->value}, bad); });
            bad = pool; bad.v[KernelParams::kAttributeSlot + 6] = 2;
            rejects([&] { backend_dispatch(op, single.node()->value, {small.node()->value}, bad); });
        }
        Tensor odd = avg_pool2d(Tensor::ones({1, 1, 1, 5}, backend, false, dtype), 1);
        if (dtype == DType::Float4E2M1FN) check((odd.raw_data().back() & 0xf0) == 0, "Packed spatial padding is dirty");
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor input, weight;
        {
            StreamScope scope(producer, asynchronous, true);
            input = Tensor::ones({1, 1, 3, 3}, backend, true, DType::Float4E2M1FN);
            weight = Tensor::ones({1, 1, 2, 2}, backend, false, DType::Float4E2M1FN);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor result = conv2d(input, weight, {});
            result.backward(Tensor::ones(result.shape(), backend, false, result.dtype()));
            check(result.data() == std::vector<float>({4, 4, 4, 4}), "Spatial stream dependency differs");
            check(Tensor(input.node()->gradient).data() == std::vector<float>({1, 2, 1, 2, 4, 2, 1, 2, 1}), "Spatial gradient stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void storage_gradients(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor x = Tensor::ones({7}, backend, true, dtype);
        Tensor selected = slice(x, 0, 1, 7, 2);
        selected.backward(Tensor::ones({3}, backend, false, dtype));
        check(Tensor(x.node()->gradient).raw_data() == Tensor::from_data({7}, {0, 1, 0, 1, 0, 1, 0}, backend, false, dtype).raw_data(),
              "Low precision slice gradient differs");
        Tensor viewed = Tensor::ones({7}, backend, true, dtype);
        Tensor view = contiguous_view(viewed, {5}, 2);
        view.backward(Tensor::ones({5}, backend, false, dtype));
        check(Tensor(viewed.node()->gradient).raw_data() == Tensor::from_data({7}, {0, 0, 1, 1, 1, 1, 1}, backend, false, dtype).raw_data(),
              "Low precision view gradient differs");
        Tensor scalar = Tensor::ones({}, backend, true, dtype);
        Tensor expanded = expand(scalar, {2, 3});
        expanded.backward(Tensor::ones({2, 3}, backend, false, dtype));
        check(Tensor(scalar.node()->gradient).raw_data() == Tensor::from_data({}, {6}, backend, false, dtype).raw_data(),
              "Low precision expand gradient rounded intermediate dimensions");
        check(squeeze(scalar, -1).shape().empty() && squeeze(scalar, 0).shape().empty(), "Scalar squeeze shape differs");
        rejects([&] { squeeze(scalar, 1); });
        rejects([&] { unsqueeze(Tensor::ones({1, 1, 1, 1, 1, 1, 1, 1}, backend, false, dtype), 0); });
        Tensor triangle = Tensor::ones({2, 3}, backend, true, dtype);
        Tensor upper = triu(triangle);
        const std::vector<uint8_t> upper_reference = Tensor::from_data({2, 3}, {1, 1, 1, 0, 1, 1}, backend, false, dtype).raw_data();
        check(upper.raw_data() == upper_reference, "Low precision triangular zero encoding differs");
        upper.backward(Tensor::ones({2, 3}, backend, false, dtype));
        check(Tensor(triangle.node()->gradient).raw_data() == upper_reference, "Low precision triangular gradient differs");
        Tensor left = Tensor::ones({3}, backend, true, dtype), right = Tensor::ones({2}, backend, true, dtype);
        Tensor joined = concat({left, right}, 0);
        joined.backward(Tensor::ones({5}, backend, false, dtype));
        check(Tensor(left.node()->gradient).raw_data() == left.raw_data() && Tensor(right.node()->gradient).raw_data() == right.raw_data(),
              "Packed concat gradient differs");
        Tensor input = Tensor::ones({3}, backend, false, dtype), output = Tensor::zeros({7}, backend, false, dtype);
        KernelParams valid = pointwise_descriptor(output, {input});
        valid.v[KernelParams::kAttributeSlot] = 0; valid.v[KernelParams::kAttributeSlot + 1] = 1; valid.v[KernelParams::kAttributeSlot + 2] = 2;
        for (const std::string op : {"slice_grad", "slice_backward"}) {
            backend_dispatch(op, output.node()->value, {input.node()->value}, valid);
            check(output.raw_data() == Tensor(x.node()->gradient).raw_data(), "Raw slice derivative alias differs");
        }
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("slice_grad", output.node()->value, {input.node()->value}, p); }};
        const std::vector<uint8_t> unchanged = output.raw_data();
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 6; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputShapeSlot] = 6; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = -1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B)] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot + 1] = -1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot + 1] = 8; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot + 2] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot + 2] = INT32_MAX; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("slice_grad", output.node()->value, {}, valid); });
        rejects([&] { backend_dispatch("slice_grad", output.node()->value, {input.node()->value, input.node()->value}, valid); });
        Tensor wrong = Tensor::ones({3}, backend);
        KernelParams mixed = pointwise_descriptor(output, {wrong}); mixed.v[KernelParams::kAttributeSlot + 2] = 1;
        rejects([&] { backend_dispatch("slice_grad", output.node()->value, {wrong.node()->value}, mixed); });
        KernelParams alias = pointwise_descriptor(input, {input}); alias.v[KernelParams::kAttributeSlot + 2] = 1;
        rejects([&] { backend_dispatch("slice_grad", input.node()->value, {input.node()->value}, alias); });
        check(output.raw_data() == unchanged, "Rejected storage gradient modified output bytes");
        Tensor storage = Tensor::ones({7}, backend, false, dtype);
        Tensor offset_output = contiguous_view(storage, {5}, 2);
        Tensor offset_input = Tensor::ones({2}, backend, false, dtype);
        dispatch_tensor_into("slice_grad", offset_output, {offset_input}, {0, 0, 2});
        check(storage.raw_data() == Tensor::from_data({7}, {1, 1, 1, 0, 1, 0, 0}, backend, false, dtype).raw_data(),
              "Offset storage gradient modified surrounding bytes");
        if (dtype == DType::Float4E2M1FN) {
            check((Tensor(x.node()->gradient).raw_data().back() & 0xf0) == 0, "Packed slice gradient padding is dirty");
            check((storage.raw_data().back() & 0xf0) == 0, "Packed offset gradient padding is dirty");
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor input;
        {
            StreamScope scope(producer, asynchronous, true);
            input = Tensor::ones({7}, backend, true, DType::Float4E2M1FN);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor selected = slice(input, 0, 1, 7, 2);
            selected.backward(Tensor::ones({3}, backend, false, DType::Float4E2M1FN));
            check(Tensor(input.node()->gradient).data() == std::vector<float>({0, 1, 0, 1, 0, 1, 0}), "Storage gradient stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void indexings(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor x = Tensor::from_data({2, 3}, {4, 1, 2, 4, 1, 2}, backend, true, dtype);
        std::pair<Tensor, Tensor> sorted = sort(x, 1);
        check(sorted.second.int_data() == std::vector<int32_t>({1, 2, 0, 1, 2, 0}), "Low precision stable sort indices differ");
        check(sorted.first.raw_data() == gather(x, 1, sorted.second).raw_data(), "Low precision sorted values changed storage");
        sorted.first.backward(Tensor::from_data({2, 3}, {1, 2, 3, 1, 2, 3}, backend, false, dtype));
        check(Tensor(x.node()->gradient).raw_data() == Tensor::from_data({2, 3}, {3, 1, 2, 3, 1, 2}, backend, false, dtype).raw_data(),
              "Low precision sorting gradient differs");
        Tensor repeated = Tensor::ones({2, 3}, backend, true, dtype);
        Tensor tiled = repeat(repeated, {2, 3});
        tiled.backward(Tensor::ones(tiled.shape(), backend, false, dtype));
        check(Tensor(repeated.node()->gradient).raw_data() == Tensor::from_data({2, 3}, std::vector<float>(6, 6), backend, false, dtype).raw_data(),
              "Low precision repeated gradient differs");
        Tensor values = Tensor::ones({2, 3}, backend, false, dtype);
        Tensor ids = Tensor::from_int_data({2, 3}, {0, 1, 0, 1, 2, 1}, backend);
        Tensor output = Tensor::zeros({2, 3}, backend, false, dtype);
        KernelParams valid = pointwise_descriptor(output, {values, ids});
        valid.v[KernelParams::kAttributeSlot] = 1;
        for (const std::string op : {"gather_grad", "gather_backward"}) {
            backend_dispatch(op, output.node()->value, {values.node()->value, ids.node()->value}, valid);
            check(output.raw_data() == Tensor::from_data({2, 3}, {2, 1, 0, 0, 2, 1}, backend, false, dtype).raw_data(),
                  "Low precision gather derivative differs");
        }
        LocalFunction raw{[&](const KernelParams& descriptor) {
            backend_dispatch("gather_grad", output.node()->value, {values.node()->value, ids.node()->value}, descriptor);
        }};
        const std::vector<uint8_t> unchanged = output.raw_data();
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 5; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputShapeSlot] = 3; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputDTypeSlot] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B)] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_dtype_slot(KernelInput::A)] = int(DType::Float32); rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::C)] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = -1; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("gather_grad", output.node()->value, {values.node()->value}, valid); });
        rejects([&] { backend_dispatch("gather_grad", output.node()->value, {values.node()->value, ids.node()->value, values.node()->value}, valid); });
        Tensor float_ids = Tensor::ones({2, 3}, backend);
        KernelParams mixed = pointwise_descriptor(output, {values, float_ids}); mixed.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("gather_grad", output.node()->value, {values.node()->value, float_ids.node()->value}, mixed); });
        Tensor out_of_range = Tensor::from_int_data({2, 3}, {0, 1, 3, 0, 1, 2}, backend);
        KernelParams invalid = pointwise_descriptor(output, {values, out_of_range}); invalid.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("gather_grad", output.node()->value, {values.node()->value, out_of_range.node()->value}, invalid); });
        rejects([&] { gather(values, 1, out_of_range); });
        KernelParams alias = pointwise_descriptor(values, {values, ids}); alias.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("gather_grad", values.node()->value, {values.node()->value, ids.node()->value}, alias); });
        rejects([&] { backend_dispatch("sort_backward", output.node()->value, {values.node()->value, ids.node()->value}, valid); });
        KernelParams repeat_bad = pointwise_descriptor(output, {Tensor::ones({2, 4}, backend, false, dtype)});
        Tensor repeat_input = Tensor::ones({2, 4}, backend, false, dtype);
        rejects([&] { backend_dispatch("repeat_grad", output.node()->value, {repeat_input.node()->value}, repeat_bad); });
        check(output.raw_data() == unchanged, "Rejected low precision indexing wrote output bytes");
        std::vector<uint8_t> codes(256);
        for (int i = 0; i < 256; ++i) codes[i] = uint8_t(i);
        const int count = dtype == DType::Float4E2M1FN ? 512 : 256;
        Tensor encoded = Tensor::from_raw({count}, dtype, codes, backend);
        std::vector<int32_t> identity(count);
        for (int i = 0; i < count; ++i) identity[i] = i;
        Tensor identity_ids = Tensor::from_int_data({count}, identity, backend);
        for (const std::string op : {"scatter_source_grad", "sort_backward"})
            check(dispatch_tensor(op, encoded.shape(), {encoded, identity_ids}, {0}, dtype).raw_data() == codes,
                  "Indexed copy changed NaN payloads or signed zero");
        check(scatter(encoded, 0, identity_ids, encoded).raw_data() == codes, "Scatter changed encoded values");
        Tensor odd = Tensor::ones({3}, backend, true, dtype);
        Tensor odd_ids = Tensor::from_int_data({3}, {2, 0, 2}, backend);
        Tensor selected = gather(odd, 0, odd_ids);
        selected.backward(Tensor::ones({3}, backend, false, dtype));
        if (dtype == DType::Float4E2M1FN) {
            check((selected.raw_data().back() & 0xf0) == 0, "Packed indexing output padding is dirty");
            check((Tensor(odd.node()->gradient).raw_data().back() & 0xf0) == 0, "Packed indexing gradient padding is dirty");
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor encoded, ids;
        {
            StreamScope scope(producer, asynchronous, true);
            encoded = Tensor::from_data({5}, {1, 2, 3, 4, 6}, backend, true, DType::Float4E2M1FN);
            ids = Tensor::from_int_data({3}, {4, 0, 2}, backend);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            Tensor selected = gather(encoded, 0, ids);
            selected.backward(Tensor::ones({3}, backend, false, DType::Float4E2M1FN));
            check(selected.data() == std::vector<float>({6, 1, 3}), "Indexed stream dependency differs");
            check(Tensor(encoded.node()->gradient).data() == std::vector<float>({1, 0, 1, 0, 1}), "Indexed gradient stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void matrices(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor a = Tensor::ones({3, 3}, backend, true, dtype);
        Tensor b = Tensor::ones({3, 5}, backend, true, dtype);
        Tensor product = matmul(a, b);
        Tensor expected = Tensor::from_data({3, 5}, std::vector<float>(15, 3.f), backend, false, dtype);
        check(product.raw_data() == expected.raw_data(), "Low precision matrix product differs");
        product.backward(Tensor::ones(product.shape(), backend, false, dtype));
        check(Tensor(a.node()->gradient).raw_data() == Tensor::from_data(a.shape(), std::vector<float>(9, 5.f), backend, false, dtype).raw_data(),
              "Low precision matrix input gradient differs");
        check(Tensor(b.node()->gradient).raw_data() == Tensor::from_data(b.shape(), std::vector<float>(15, 3.f), backend, false, dtype).raw_data(),
              "Low precision matrix weight gradient differs");
        Tensor w = Tensor::ones({5, 3}, backend, true, dtype);
        Tensor x = Tensor::ones({3, 3}, backend, true, dtype);
        Tensor bias = Tensor::zeros({5}, backend, true, dtype);
        // E8M0 has no zero; retain its decoded bias in the arithmetic reference.
        Tensor reference = cast(linear(cast(x.detach(), DType::Float32), cast(w.detach(), DType::Float32), cast(bias.detach(), DType::Float32)), dtype);
        check(linear(x, w, bias).raw_data() == reference.raw_data(), "Low precision linear bias differs");
        const std::vector<int> widths{1, 0, 3, 1};
        Tensor packed = packed_linear(x, w, widths);
        packed.backward(Tensor::ones(packed.shape(), backend, false, dtype));
        check(packed.raw_data() == reshape(expected, {15}).raw_data(), "Packed low precision matrix layout differs");
        check(Tensor(x.node()->gradient).raw_data() == Tensor::from_data(x.shape(), std::vector<float>(9, 5.f), backend, false, dtype).raw_data(),
              "Packed input gradients rounded before accumulation");
        check(Tensor(w.node()->gradient).raw_data() == Tensor::from_data(w.shape(), std::vector<float>(15, 3.f), backend, false, dtype).raw_data(),
              "Packed matrix weight gradient differs");
        if (dtype == DType::Float4E2M1FN) check((packed.raw_data().back() & 0xf0) == 0, "Packed matrix padding is not zero");
        Tensor output = Tensor::zeros({3, 5}, backend, false, dtype);
        const std::vector<uint8_t> unchanged = output.raw_data();
        const KernelParams valid = pointwise_descriptor(output, {a, b});
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("matmul", output.node()->value, {a.node()->value, b.node()->value}, p); }};
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 14; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputShapeSlot] = 5; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B) + KernelParams::kInputShapeOffset] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot + 1] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::C)] = 0; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::D)] = 0; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("matmul", output.node()->value, {a.node()->value}, valid); });
        rejects([&] { backend_dispatch("matmul", output.node()->value, {a.node()->value, b.node()->value, a.node()->value}, valid); });
        Tensor wrong = Tensor::ones(b.shape(), backend);
        KernelParams mixed = pointwise_descriptor(output, {a, wrong});
        rejects([&] { backend_dispatch("matmul", output.node()->value, {a.node()->value, wrong.node()->value}, mixed); });
        check(output.raw_data() == unchanged, "Rejected matrix descriptor wrote output bytes");
        KernelParams alias = pointwise_descriptor(b, {a, b});
        rejects([&] { backend_dispatch("matmul", b.node()->value, {a.node()->value, b.node()->value}, alias); });
        check(b.data() == std::vector<float>(15, 1.f), "Rejected matrix alias changed storage");
        KernelParams raw_bmm = pointwise_descriptor(output, {a, b});
        rejects([&] { backend_dispatch("bmm", output.node()->value, {a.node()->value, b.node()->value}, raw_bmm); });
        Tensor flat = Tensor::zeros({15}, backend, false, dtype);
        KernelParams projection = pointwise_descriptor(flat, {a, w});
        projection.v[KernelParams::kAttributeSlot] = 3;
        projection.v[KernelParams::kAttributeSlot + 1] = 1;
        projection.v[KernelParams::kAttributeSlot + 2] = 0;
        projection.v[KernelParams::kAttributeSlot + 3] = 4;
        backend_dispatch("packed_linear", flat.node()->value, {a.node()->value, w.node()->value}, projection);
        check(flat.raw_data() == reshape(expected, {15}).raw_data(), "Raw packed projection differs");
        for (std::pair<int, int> entry : std::vector<std::pair<int,int>>{{0,0},{0,64},{1,-1},{3,5}}) {
            bad = projection; bad.v[KernelParams::kAttributeSlot + entry.first] = entry.second;
            rejects([&] { backend_dispatch("packed_linear", flat.node()->value, {a.node()->value, w.node()->value}, bad); });
        }
        Tensor addend = Tensor::ones({5}, backend, false, dtype);
        KernelParams addition = pointwise_descriptor(output, {a, b, addend});
        const float one = 1.f;
        std::memcpy(&addition.v[KernelParams::kAttributeSlot], &one, sizeof(one));
        std::memcpy(&addition.v[KernelParams::kAttributeSlot + 1], &one, sizeof(one));
        backend_dispatch("addmm", output.node()->value, {a.node()->value, b.node()->value, addend.node()->value}, addition);
        check(output.data() == std::vector<float>(15, 4.f), "Raw low precision addmm differs");
        rejects([&] { backend_dispatch("addmm", output.node()->value, {a.node()->value, b.node()->value}, addition); });
        Tensor mutable_input = Tensor::ones({2, 3}, backend, true, dtype);
        Tensor tracked = linear(mutable_input, w.detach());
        mutable_input.assign(std::vector<float>(6, 2.f));
        rejects([&] { tracked.backward(Tensor::ones(tracked.shape(), backend, false, dtype)); });
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor input, weight, result;
        {
            StreamScope scope(producer, asynchronous, true);
            input = Tensor::ones({3, 3}, backend, true, DType::Float4E2M1FN);
            weight = Tensor::ones({5, 3}, backend, true, DType::Float4E2M1FN);
            result = packed_linear(input, weight, {1, 0, 3, 1});
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            check(result.data() == std::vector<float>(15, 3.f), "Matrix producer stream was lost");
            result.backward(Tensor::ones(result.shape(), backend, false, result.dtype()));
            check(input.grad() == std::vector<float>(9, 4.f), "Async packed matrix gradient differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void softmaxes(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        for (const std::string operation : {"softmax", "log_softmax"}) {
            const bool logarithmic = operation == "log_softmax";
            Tensor input = Tensor::ones({2, 2}, backend, true, dtype);
            Tensor result = logarithmic ? log_softmax(input, 1) : softmax(input, 1);
            const float forward = logarithmic ? -std::log(2.f) : .5f;
            check(result.raw_data() == Tensor::from_data({2, 2}, std::vector<float>(4, forward), backend, false, dtype).raw_data(), "Uniform low precision softmax differs");
            result.backward(Tensor::from_data({2, 2}, {1.f, 0.f, 0.f, 1.f}, backend));
            const float derivative = logarithmic ? .5f : .25f;
            Tensor expected = Tensor::from_data({2, 2}, {derivative, -derivative, -derivative, derivative}, backend, false, dtype);
            check(Tensor(input.node()->gradient).raw_data() == expected.raw_data(), "Low precision softmax Jacobian differs");
            input.zero_grad();
            result.backward(Tensor::from_data({2, 2}, {1.f, 0.f, 0.f, 1.f}, backend));
            check(Tensor(input.node()->gradient).raw_data() == expected.raw_data(), "Repeated softmax backward lost saved probabilities");
            Tensor output = Tensor::zeros(input.shape(), backend, false, dtype);
            KernelParams valid = descriptor(output, input); valid.v[KernelParams::kAttributeSlot] = 1;
            LocalFunction raw{[&](const KernelParams& params) { backend_dispatch(operation, output.node()->value, {input.node()->value}, params); }};
            raw(valid); check(output.raw_data() == result.raw_data(), "Raw and tracked softmax differ");
            const std::vector<uint8_t> unchanged = output.raw_data();
            for (int field : {KernelParams::kOutputCountSlot, KernelParams::kOutputRankSlot, KernelParams::kOutputShapeSlot,
                               KernelParams::input_slot(KernelInput::A), KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset}) {
                KernelParams bad = valid; bad.v[field] = 9; rejects([&] { raw(bad); });
            }
            for (int axis : {-1, 2}) { KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot] = axis; rejects([&] { raw(bad); }); }
            for (int mode : {-1, 1, 2, 3}) { KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot + 1] = mode; rejects([&] { raw(bad); }); }
            for (elman::KernelInput descriptor_input : {KernelInput::B, KernelInput::C, KernelInput::D}) {
                KernelParams bad = valid; bad.v[KernelParams::input_slot(descriptor_input)] = 0; rejects([&] { raw(bad); });
            }
            rejects([&] { backend_dispatch(operation, output.node()->value, {}, valid); });
            rejects([&] { backend_dispatch(operation, output.node()->value, {input.node()->value, input.node()->value}, valid); });
            check(output.raw_data() == unchanged, "Rejected softmax modified output");
            Tensor wide = dispatch_tensor(operation, input.shape(), {input}, {1, 1}, DType::Float32);
            Tensor upstream = Tensor::ones(input.shape(), backend, false, dtype);
            KernelParams backward = pointwise_descriptor(output, {wide, upstream});
            backward.v[KernelParams::kAttributeSlot] = 1; backward.v[KernelParams::kAttributeSlot + 1] = 2;
            KernelParams bad = backward; bad.v[KernelParams::input_slot(KernelInput::B) + KernelParams::kInputStrideOffset] = 9;
            rejects([&] { backend_dispatch(operation, output.node()->value, {wide.node()->value, upstream.node()->value}, bad); });
            rejects([&] { backend_dispatch(operation, upstream.node()->value, {wide.node()->value, upstream.node()->value}, backward); });
            Tensor inplace = Tensor::ones({3}, backend, false, dtype);
            Tensor fresh = logarithmic ? log_softmax(inplace) : softmax(inplace);
            dispatch_tensor_into(operation, inplace, {inplace}, {0});
            check(inplace.raw_data() == fresh.raw_data(), "In-place softmax differs");
            if (dtype == DType::Float4E2M1FN) check((inplace.raw_data().back() & 0xf0) == 0, "Softmax packed padding is nonzero");
            Tensor storage = Tensor::ones({10}, backend, false, dtype);
            Tensor source = contiguous_view(storage, {8}), destination = contiguous_view(storage, {8}, 2);
            KernelParams overlap = descriptor(destination, source);
            rejects([&] { backend_dispatch(operation, destination.node()->value, {source.node()->value}, overlap); });
            input.assign(std::vector<float>(4, 2.f));
            rejects([&] { result.backward(Tensor::ones(result.shape(), backend)); });
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor input, result;
        {
            StreamScope scope(producer, asynchronous, true);
            input = Tensor::ones({3}, backend, true, DType::Float4E2M1FN);
            result = softmax(input);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            check(result.data() == std::vector<float>(3, .5f), "Softmax producer stream was lost");
            result.backward(Tensor::ones(result.shape(), backend));
            check(input.grad() == std::vector<float>(3, 0.f), "Async softmax gradient differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void normalizations(Backend backend) {
    Tensor (*const normalize)(const std::string&, const Tensor&, const Tensor&, const Tensor&) = [](const std::string& op, const Tensor& x, const Tensor& w, const Tensor& b) {
        if (op == "layer_norm") return layer_norm(x, w, b, 0.);
        if (op == "rms_norm") return rms_norm(x, w, 0.);
        if (op == "batch_norm") return batch_norm(x, w, b, 0.);
        return group_norm(x, 1, w, b, 0.);
    };
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        for (const std::string op : {"layer_norm", "rms_norm", "batch_norm", "group_norm"}) {
            const bool rms = op == "rms_norm", batch = op == "batch_norm";
            Tensor x = Tensor::from_data({3, 3}, {.5f, 1.f, 2.f, 1.f, 2.f, .5f, 2.f, .5f, 1.f}, backend, true, dtype);
            Tensor w = Tensor::ones({3}, backend, true, dtype), b = Tensor::ones({3}, backend, true, dtype);
            Tensor y = normalize(op, x, w, b);
            y.backward(Tensor::ones(y.shape(), backend, false, dtype));
            const std::vector<uint8_t> dx = Tensor(x.node()->gradient).raw_data(), dw = Tensor(w.node()->gradient).raw_data();
            x.zero_grad(); w.zero_grad(); b.zero_grad();
            y.backward(Tensor::ones(y.shape(), backend, false, dtype));
            check(dx == Tensor(x.node()->gradient).raw_data() && dw == Tensor(w.node()->gradient).raw_data(), "Normalization repeated backward differs");
            if (!rms) check(b.grad() == std::vector<float>(3, dtype == DType::Float8E8M0FNU ? 4.f : 3.f), "Normalization bias gradient differs");
            if (dtype == DType::Float4E2M1FN) {
                check((y.raw_data().back() & 0xf0) == 0, "Normalization output padding is nonzero");
                check((dx.back() & 0xf0) == 0 && (dw.back() & 0xf0) == 0, "Normalization gradient padding is nonzero");
            }
            Tensor output = Tensor::zeros(x.shape(), backend, false, dtype);
            std::vector<Tensor> operands{x, w}; if (!rms) operands.push_back(b);
            KernelParams valid = pointwise_descriptor(output, operands);
            valid.v[KernelParams::kAttributeSlot] = batch ? 0 : 1;
            valid.v[KernelParams::kAttributeSlot + 9] = batch ? 1 : 0;
            std::vector<BufferPtr> buffers; for (const elman::Tensor& value : operands) buffers.push_back(value.node()->value);
            LocalFunction raw{[&](const KernelParams& p) { backend_dispatch(op, output.node()->value, buffers, p); }};
            raw(valid); check(output.raw_data() == y.raw_data(), "Normalization raw/public paths differ");
            const std::vector<uint8_t> unchanged = output.raw_data();
            for (int field : {KernelParams::kOutputCountSlot, KernelParams::kOutputRankSlot, KernelParams::kOutputShapeSlot,
                               KernelParams::input_slot(KernelInput::A), KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset,
                               KernelParams::input_slot(KernelInput::B), KernelParams::input_slot(KernelInput::B) + KernelParams::kInputShapeOffset}) {
                KernelParams bad = valid; bad.v[field] = 99; rejects([&] { raw(bad); });
            }
            for (int mode : {-1, 4}) { KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot + 8] = mode; rejects([&] { raw(bad); }); }
            KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot + 2] = 2; rejects([&] { raw(bad); });
            bad = valid; bad.v[KernelParams::kAttributeSlot + (batch ? 0 : 1)] = int(0xbf800000u); rejects([&] { raw(bad); });
            bad = valid; bad.v[KernelParams::input_slot(KernelInput::D)] = 0; rejects([&] { raw(bad); });
            rejects([&] { backend_dispatch(op, output.node()->value, {}, valid); });
            rejects([&] { backend_dispatch(op, x.node()->value, buffers, valid); });
            check(output.raw_data() == unchanged, "Rejected normalization modified output");
            Tensor storage = Tensor::ones({14}, backend, false, dtype);
            Tensor source = contiguous_view(storage, {4, 3}), destination = contiguous_view(storage, {4, 3}, 2);
            std::vector<Tensor> overlapping = operands; overlapping[0] = source;
            KernelParams overlap = pointwise_descriptor(destination, overlapping); overlap.v[KernelParams::kAttributeSlot] = batch ? 0 : 1;
            overlap.v[KernelParams::kAttributeSlot + 9] = batch ? 1 : 0;
            std::vector<BufferPtr> overlap_buffers = buffers; overlap_buffers[0] = source.node()->value;
            rejects([&] { backend_dispatch(op, destination.node()->value, overlap_buffers, overlap); });
            for (int parent = 0; parent < (rms ? 2 : 3); ++parent) {
                Tensor fresh_x = Tensor::from_raw(x.shape(), dtype, x.raw_data(), backend, true);
                Tensor fresh_w = Tensor::from_raw(w.shape(), dtype, w.raw_data(), backend, true);
                Tensor fresh_b = Tensor::from_raw(b.shape(), dtype, b.raw_data(), backend, true);
                Tensor result = normalize(op, fresh_x, fresh_w, fresh_b);
                Tensor changed = parent == 0 ? fresh_x : parent == 1 ? fresh_w : fresh_b;
                changed.assign(std::vector<float>(changed.numel(), 2.f));
                rejects([&] { result.backward(Tensor::ones(result.shape(), backend)); });
            }
        }
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor input, weight, result;
        {
            StreamScope scope(producer, asynchronous, true);
            input = Tensor::from_data({3}, {.5f, 1.f, 2.f}, backend, true, DType::Float4E2M1FN);
            weight = Tensor::from_data({3}, {.5f, 2.f, 4.f}, backend, true, DType::Float4E2M1FN);
            result = rms_norm(input, weight, 0., RmsNormRounding::Normalized);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            check(result.data() == std::vector<float>({0.f, 2.f, 6.f}), "Normalization producer stream was lost");
            check(rms_norm(input, weight, 0.).data() == std::vector<float>({0.f, 1.5f, 6.f}), "RMS intermediate rounding boundary was lost");
            Tensor raw = dispatch_tensor("rms_norm", input.shape(), {input, weight}, {1, 0, 1}, input.dtype());
            check(raw.raw_data() == result.raw_data(), "Raw normalized RMS differs from the public operation");
            result.backward(Tensor::ones(result.shape(), backend));
            check(weight.grad() == std::vector<float>({.5f, 1.f, 1.5f}), "Async normalization affine gradient differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void raw_normalizations(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor x = Tensor::ones({3, 3}, backend, false, dtype), w = Tensor::ones({3}, backend, false, dtype);
        Tensor b = Tensor::zeros({3}, backend, false, dtype), g = Tensor::ones(x.shape(), backend, false, dtype);
        for (const std::string op : {"layer_norm", "rms_norm", "group_norm", "batch_norm", "group_norm_input_grad", "group_norm_weight_grad"}) {
            const bool batch = op == "batch_norm", dw = op == "group_norm_weight_grad", dx = op == "group_norm_input_grad", rms = op == "rms_norm";
            std::vector<Tensor> inputs = batch ? std::vector<Tensor>{x, w, w} : dx ? std::vector<Tensor>{x, w, g} : dw ? std::vector<Tensor>{x, g} : std::vector<Tensor>{x};
            std::vector<int> attributes{batch ? int(0x3f800000) : 1, int(0x3f800000)};
            Tensor result = dispatch_tensor(op, dw ? w.shape() : x.shape(), inputs, attributes, dtype);
            Tensor wanted = rms ? Tensor::from_data(x.shape(), std::vector<float>(9, 1.f / std::sqrt(2.f)), backend, false, dtype) : Tensor::zeros(result.shape(), backend, false, dtype);
            check(result.raw_data() == wanted.raw_data(), "Raw normalization constant reference differs");
            if (dtype == DType::Float4E2M1FN) check((result.raw_data().back() & 0xf0) == 0, "Raw normalization packed padding is nonzero");
            Tensor output = Tensor::zeros(result.shape(), backend, false, dtype);
            KernelParams valid = pointwise_descriptor(output, inputs);
            for (size_t i = 0; i < attributes.size(); ++i) valid.v[KernelParams::kAttributeSlot + i] = attributes[i];
            std::vector<BufferPtr> buffers; for (const elman::Tensor& input : inputs) buffers.push_back(input.node()->value);
            LocalFunction dispatch{[&](const KernelParams& p) { backend_dispatch(op, output.node()->value, buffers, p); }};
            dispatch(valid); check(output.raw_data() == result.raw_data(), "Raw normalization descriptor differs");
            const std::vector<uint8_t> unchanged = output.raw_data();
            for (int slot : {KernelParams::kOutputCountSlot, KernelParams::kOutputRankSlot, KernelParams::kOutputShapeSlot,
                             KernelParams::input_slot(KernelInput::A), KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset}) {
                KernelParams bad = valid; bad.v[slot] = 99; rejects([&] { dispatch(bad); });
            }
            for (int attribute : {2, 7, 9}) { KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot + attribute] = 2; rejects([&] { dispatch(bad); }); }
            for (int mode : {-1, 4}) { KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot + 8] = mode; rejects([&] { dispatch(bad); }); }
            KernelParams bad = valid; bad.v[KernelParams::kAttributeSlot + 7] = 1; bad.v[KernelParams::kAttributeSlot + 6] = int(0xbff00000); rejects([&] { dispatch(bad); });
            bad.v[KernelParams::kAttributeSlot + 6] = int(0x7ff80000); rejects([&] { dispatch(bad); });
            bad = valid; bad.v[KernelParams::input_slot(KernelInput::D)] = 0; rejects([&] { dispatch(bad); });
            rejects([&] { backend_dispatch(op, output.node()->value, {}, valid); });
            if (dx || dw || batch) {
                bad = valid; bad.v[KernelParams::kAttributeSlot + 8] = 1; rejects([&] { dispatch(bad); });
                std::vector<BufferPtr> missing = buffers; missing[1].reset(); rejects([&] { backend_dispatch(op, output.node()->value, missing, valid); });
            }
            if (!dw) rejects([&] { backend_dispatch(op, x.node()->value, buffers, valid); });
            check(output.raw_data() == unchanged, "Rejected raw normalization modified output");
            if (!batch && !dx && !dw) {
                Tensor bias_only = dispatch_tensor(op, x.shape(), {x, Tensor{}, b}, attributes, dtype);
                check(bias_only.raw_data() == result.raw_data(), "Optional normalization weight was not skipped");
            }
        }
        Tensor input = Tensor::from_data({2, 3}, {1.f, 2.f, 3.f, 1.f, 2.f, 3.f}, backend, false, dtype);
        Tensor variance = Tensor::from_data({3}, {4.f, 4.f, 4.f}, backend, false, dtype);
        Tensor inference = dispatch_tensor("batch_norm", input.shape(), {input, w, variance}, {0}, dtype);
        Tensor expected = Tensor::from_data(input.shape(), {0.f, .5f, 1.f, 0.f, .5f, 1.f}, backend, false, dtype);
        if (dtype != DType::Float8E8M0FNU) check(inference.raw_data() == expected.raw_data(), "Inference BatchNorm used training statistics");
        // E8M0 rounds the value three to four before normalization.
        else check(inference.raw_data() == Tensor::from_data(input.shape(), {0.f, .5f, 2.f, 0.f, .5f, 2.f}, backend, false, dtype).raw_data(), "E8M0 inference BatchNorm differs");
        check(batch_norm(input, w, variance, 1.).raw_data() == Tensor::from_data(input.shape(), std::vector<float>(6, 4.f), backend, false, dtype).raw_data(), "Training BatchNorm used inference statistics");
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor result;
        {
            StreamScope scope(producer, asynchronous, true);
            Tensor x = Tensor::from_data({1, 3}, {1.f, 2.f, 3.f}, backend, false, DType::Float4E2M1FN);
            Tensor m = Tensor::ones({3}, backend, false, x.dtype());
            Tensor v = Tensor::from_data({3}, {4.f, 4.f, 4.f}, backend, false, x.dtype());
            result = dispatch_tensor("batch_norm", x.shape(), {x, m, v}, {0}, x.dtype());
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            check(result.data() == std::vector<float>({0.f, .5f, 1.f}), "Raw normalization producer lifetime was lost");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void reductions(Backend backend) {
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor input = Tensor::from_data({2, 3}, {.5f, 1.f, 1.5f, 2.f, 3.f, 4.f}, backend, true, dtype);
        Tensor wide = cast(input.detach(), DType::Float32);
#define ELMAN_LOW_CHECK_REDUCTION(name) check(name(input, 1).raw_data() == cast(name(wide, 1), dtype).raw_data(), "Low precision " #name " differs")
        ELMAN_LOW_CHECK_REDUCTION(sum); ELMAN_LOW_CHECK_REDUCTION(mean); ELMAN_LOW_CHECK_REDUCTION(prod);
        ELMAN_LOW_CHECK_REDUCTION(max); ELMAN_LOW_CHECK_REDUCTION(min); ELMAN_LOW_CHECK_REDUCTION(amax); ELMAN_LOW_CHECK_REDUCTION(amin);
#undef ELMAN_LOW_CHECK_REDUCTION
        Tensor loss = sum_all(input);
        loss.backward(); loss.backward();
        check(input.grad() == std::vector<float>(6, 2.f), "Low precision reduction gradient accumulation differs");
        Tensor output = Tensor::zeros({2}, backend, false, dtype);
        const std::vector<uint8_t> untouched = output.raw_data();
        KernelParams valid = descriptor(output, input);
        valid.v[KernelParams::kAttributeSlot] = 1;
        LocalFunction raw{[&](const KernelParams& p) { backend_dispatch("sum", output.node()->value, {input.node()->value}, p); }};
        KernelParams bad = valid; bad.v[KernelParams::kOutputCountSlot] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputRankSlot] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kOutputShapeSlot] = 3; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A)] = 9; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputShapeOffset] = -1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset] = 1; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot] = -2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::kAttributeSlot + 1] = 2; rejects([&] { raw(bad); });
        bad = valid; bad.v[KernelParams::input_slot(KernelInput::B)] = 0; rejects([&] { raw(bad); });
        rejects([&] { backend_dispatch("sum", output.node()->value, {}, valid); });
        rejects([&] { backend_dispatch("sum", output.node()->value, {input.node()->value, input.node()->value}, valid); });
        check(output.raw_data() == untouched, "Rejected reduction wrote output bytes");
        Tensor wrong = Tensor::zeros({2}, backend);
        KernelParams wrong_descriptor = descriptor(wrong, input); wrong_descriptor.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("sum", wrong.node()->value, {input.node()->value}, wrong_descriptor); });
        Tensor overlapping = contiguous_view(input, {2});
        KernelParams overlap = descriptor(overlapping, input); overlap.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("sum", overlapping.node()->value, {input.node()->value}, overlap); });
        KernelParams scan = descriptor(input, input); scan.v[KernelParams::kAttributeSlot] = 1;
        rejects([&] { backend_dispatch("cumsum", input.node()->value, {input.node()->value}, scan); });
        rejects([&] { max(Tensor::zeros({2, 0}, backend, false, dtype), 1); });
        rejects([&] { cumsum(input, 2); });
        Tensor saved = sum(input.detach(), 1), upstream = Tensor::ones({2}, backend, false, dtype);
        Tensor gradient = Tensor::zeros(input.shape(), backend);
        KernelParams gradient_descriptor = pointwise_descriptor(gradient, {input, saved, upstream});
        gradient_descriptor.v[KernelParams::kAttributeSlot] = 1;
        backend_dispatch("reduce_grad", gradient.node()->value, {input.node()->value, saved.node()->value, upstream.node()->value}, gradient_descriptor);
        check(gradient.data() == std::vector<float>(6, 1.f), "Raw low precision gradient differs");
        bad = gradient_descriptor; bad.v[KernelParams::kAttributeSlot + 2] = 7;
        rejects([&] { backend_dispatch("reduce_grad", gradient.node()->value, {input.node()->value, saved.node()->value, upstream.node()->value}, bad); });
        rejects([&] { backend_dispatch("reduce_grad", gradient.node()->value, {input.node()->value, saved.node()->value}, gradient_descriptor); });
        Tensor negative_zero = Tensor::from_data({}, {-0.f}, backend, false, dtype);
        Tensor scalar_leaf = Tensor::ones({}, backend, true, dtype);
        amin(scalar_leaf, 0).backward(negative_zero);
        check(Tensor(scalar_leaf.node()->gradient).raw_data() == Tensor::zeros({}, backend, false, dtype).raw_data(),
              "Scalar amin backward zero sign differs");
        Tensor odd = Tensor::ones({3, 5}, backend, true, dtype);
        Tensor prefix = cumsum(odd, 1);
        check(prefix.raw_data() == cast(cumsum(cast(odd.detach(), DType::Float32), 1), dtype).raw_data(), "Packed scan forward differs");
        prefix.backward(Tensor::ones(prefix.shape(), backend, false, dtype));
        check(Tensor(odd.node()->gradient).raw_data() == Tensor::from_data({3, 5}, {5,4,3,2,1,5,4,3,2,1,5,4,3,2,1}, backend, false, dtype).raw_data(),
              "Packed scan backward differs");
        if (dtype == DType::Float4E2M1FN) check((prefix.raw_data().back() & 0xf0) == 0, "Scan left dirty FP4 padding");
    }
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        Tensor scanned;
        {
            StreamScope scope(producer, asynchronous, true);
            scanned = cumsum(Tensor::ones({3, 1025}, backend, false, DType::Float4E2M1FN), 1);
        }
        {
            StreamScope scope(consumer, asynchronous, true);
            check(min(scanned, 1).data() == std::vector<float>(3, 1.f), "Scan scratch or stream dependency differs");
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void pointwise(Backend backend) {
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16, DType::Float64}) {
        Tensor values = Tensor::from_data({4}, {1.f, -1.f, 0.f, std::numeric_limits<float>::quiet_NaN()}, backend, true, dtype);
        sign(values).backward(Tensor::from_data({4}, {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::quiet_NaN(), -1.f}, backend, false, dtype));
        const std::vector<uint8_t> gradient = Tensor(values.node()->gradient).raw_data();
        check(std::all_of(gradient.begin(), gradient.end(), [](uint8_t byte) { return byte == 0; }),
              "Sign backward must return positive zero independently of its upstream");
    }
    for (elman::DType dtype : {DType::Float8E4M3FN, DType::Float8E4M3FNUZ, DType::Float8E5M2,
                       DType::Float8E5M2FNUZ, DType::Float8E8M0FNU, DType::Float4E2M1FN}) {
        Tensor a = Tensor::ones({7}, backend, true, dtype);
        Tensor b = Tensor::ones({7}, backend, false, dtype);
        check(add(a, b).data() == std::vector<float>(7, 2.f), "Low precision addition differs");
        Tensor loss = sum(cast(mul(a, b), DType::Float32), 0);
        loss.backward();
        check(a.grad() == std::vector<float>(7, 1.f), "Low precision multiplication gradient differs");
        loss.backward();
        check(a.grad() == std::vector<float>(7, 2.f), "Repeated low precision backward did not accumulate");
        a.zero_grad();
        sum(add(cast(a, DType::Float32), cast(a, DType::Float32)), 0).backward();
        check(a.grad() == std::vector<float>(7, 2.f), "Branched low precision cast gradient differs");
        Tensor leaf = Tensor::from_data({1}, {1.1f}, backend, true);
        Tensor roundtrip = cast(cast(leaf, dtype), DType::Float32);
        check(roundtrip.requires_grad(), "Low precision cast detached its input");
        roundtrip.backward(Tensor::from_data({1}, {.3f}, backend));
        const float expected = dtype == DType::Float4E2M1FN ? .5f : dtype == DType::Float8E8M0FNU ? .25f : .3125f;
        check(leaf.grad() == std::vector<float>{expected}, "Cast backward omitted intermediate dtype rounding");
        Tensor output = add(a, b);
        a.assign(std::vector<float>(7, 2.f));
        rejects([&] { output.backward(Tensor::ones({7}, backend)); });
        rejects([&] { add(a, Tensor::ones({7}, backend)); });
        Tensor target = Tensor::zeros({7}, backend, false, dtype);
        const KernelParams valid = pointwise_descriptor(target, {a, b});
        const std::vector<uint8_t> untouched = target.raw_data();
        for (int field : {KernelParams::kOutputCountSlot, KernelParams::kOutputRankSlot, KernelParams::kOutputShapeSlot,
                          KernelParams::input_slot(KernelInput::A),
                          KernelParams::input_slot(KernelInput::A) + KernelParams::kInputStrideOffset,
                          KernelParams::input_slot(KernelInput::B)}) {
            KernelParams bad = valid; bad.v[field] = 9;
            rejects([&] { backend_dispatch("add", target.node()->value, {a.node()->value, b.node()->value}, bad); });
        }
        rejects([&] { backend_dispatch("add", target.node()->value, {a.node()->value}, valid); });
        rejects([&] { backend_dispatch("add", target.node()->value, {a.node()->value, b.node()->value, b.node()->value}, valid); });
        check(target.raw_data() == untouched, "Rejected pointwise descriptor wrote output bytes");
        Tensor wide = Tensor::ones({7}, backend);
        KernelParams wrong_type = pointwise_descriptor(target, {a, wide});
        rejects([&] { backend_dispatch("add", target.node()->value, {a.node()->value, wide.node()->value}, wrong_type); });
        KernelParams bad_where = pointwise_descriptor(target, {a, a, b});
        rejects([&] { backend_dispatch("where", target.node()->value, {a.node()->value, a.node()->value, b.node()->value}, bad_where); });
        Tensor storage = Tensor::ones({10}, backend, false, dtype);
        Tensor source_view = contiguous_view(storage, {8});
        Tensor output_view = contiguous_view(storage, {8}, 2);
        Tensor other = Tensor::ones({8}, backend, false, dtype);
        KernelParams overlap = pointwise_descriptor(output_view, {source_view, other});
        rejects([&] { backend_dispatch("add", output_view.node()->value, {source_view.node()->value, other.node()->value}, overlap); });
        check(storage.data() == std::vector<float>(10, 1.f), "Overlapping pointwise rejection changed storage");
        Tensor inplace = Tensor::ones({4097}, backend, false, dtype);
        Tensor scalar = Tensor::ones({}, backend, false, dtype);
        dispatch_tensor_into("add", inplace, {inplace, scalar});
        check(inplace.data() == std::vector<float>(4097, 2.f), "In-place packed pointwise execution differs");
        if (dtype == DType::Float4E2M1FN) check((inplace.raw_data().back() & 0xf0) == 0, "Pointwise packed padding is not zero");
        std::vector<uint8_t> first_codes(256), second_codes(256), expected_codes(256);
        for (int i = 0; i < 256; ++i) {
            first_codes[i] = uint8_t(i); second_codes[i] = uint8_t(255 - i);
            expected_codes[i] = dtype == DType::Float4E2M1FN ? uint8_t((i & 15) | ((255 - i) & 240)) :
                uint8_t(i % 2 ? 255 - i : i);
        }
        const int count = dtype == DType::Float4E2M1FN ? 512 : 256;
        std::vector<int32_t> selected(count);
        for (int i = 0; i < count; ++i) selected[i] = i % 2 == 0;
        Tensor condition = Tensor::from_int_data({count}, selected, backend, DType::Bool);
        Tensor first = Tensor::from_raw({count}, dtype, first_codes, backend);
        Tensor second = Tensor::from_raw({count}, dtype, second_codes, backend);
        check(where(condition, first, second).raw_data() == expected_codes, "Where changed selected raw NaN or packed encodings");
        dispatch_tensor_into("where", first, {condition, first, second});
        check(first.raw_data() == expected_codes, "In-place where changed selected raw encodings");
    }
}
void float16_arithmetic(Backend backend) {
    const std::vector<float> edges{
        1.f + std::ldexp(1.f, -11), 1.f + std::ldexp(3.f, -11),
        std::ldexp(1.f, -24), std::ldexp(1.f, -25), -std::ldexp(1.f, -25),
        65504.f, 65520.f
    };
    const std::vector<uint8_t> expected{
        0x00, 0x3c, 0x02, 0x3c, // Ties around one round to an even significand.
        0x01, 0x00, 0x00, 0x00, 0x00, 0x80, // Subnormal limit and signed zeros.
        0xff, 0x7b, 0x00, 0x7c // Largest finite value and overflow.
    };
    const Tensor input = Tensor::from_data({int(edges.size())}, edges, backend);
    check(cast(input, DType::Float16).raw_data() == expected, "Float16 cast boundary rounding differs");
    check(Tensor::from_data(input.shape(), edges, backend, false, DType::Float16).raw_data() == expected,
          "Float16 upload boundary rounding differs");

    Tensor x = Tensor::from_data({2, 3}, {1, 2, -1, 0, -2, 3}, backend, true, DType::Float16);
    Tensor weight = Tensor::from_data({3, 2}, {2, 1, 1, -2, -4, 3}, backend, true, DType::Float16);
    Tensor output = matmul(x, weight);
    check(output.dtype() == DType::Float16 && output.shape() == Shape{2, 2} &&
          output.data() == std::vector<float>{8, -6, -14, 13}, "Float16 matmul forward differs");
    output.backward(Tensor::from_data({2, 2}, {1, -2, 3, .5f}, backend, false, DType::Float16));
    check(x.grad() == std::vector<float>{0, 5, -10, 6.5f, 2, -10.5f}, "Float16 matmul input gradient differs");
    check(weight.grad() == std::vector<float>{1, -2, -4, -5, 8, 3.5f}, "Float16 matmul weight gradient differs");
}

struct Suite {
    const char* option;
    void (*run)(Backend);
};

const Suite float8_suites[] = {
    {"--cast", native},
    {"--pointwise", pointwise},
    {"--reductions", reductions},
    {"--matrices", matrices},
    {"--softmax", softmaxes},
    {"--fused", fusions},
    {"--attention", attention},
    {"--random", randoms},
    {"--floating-dropout", floating_dropouts},
    {"--sampling", samplings},
    {"--losses", losses},
    {"--spatial", spatials},
    {"--storage-gradients", storage_gradients},
    {"--indexing", indexings},
    {"--normalization", normalizations},
    {"--raw-normalization", raw_normalizations},
};

const Suite other_suites[] = {
    {"--float16", float16_arithmetic},
    {"--floating-random", floating_randoms},
};

void reference(Backend backend, const std::filesystem::path& path) {
    const Json suite = Json::read(path);
    int failed = 0;
    for (const elman::Json& original : suite.at("cases").array()) {
        elman::Json::Object object = original.object();
        object["backend"] = backend_name(backend);
        const Json test(std::move(object));
        try {
            std::vector<Tensor> inputs;
            for (const elman::Json& spec : test.at("inputs").array()) {
                inputs.push_back(spec.is_null() ? Tensor{} :
                    Tensor::from_raw(shape(spec), DType(spec.at("dtype").integer()), bytes(spec),
                                     backend, spec.bool_or("grad", false)));
            }
            if (test.bool_or("reject", false)) {
                rejects([&] { evaluate(test, inputs); });
                continue;
            }
            Tensor output = evaluate(test, inputs);
            const std::string name = test.at("name").string();
            const bool observed_mask = test.bool_or("observed_mask_control", false);
            const std::pair<std::vector<float>, std::vector<float>> scales = observed_mask ? observed_dropout_scales(test, inputs.at(0)) :
                std::pair<std::vector<float>, std::vector<float>>{};
            if (observed_mask) compare(output, specification(conditioned_dropout(inputs.at(0), scales.first)), name);
            else if (test.contains("distribution")) random_distribution(output, test, suite);
            else compare(output, test.at("expected"), name);
            if (test.bool_or("replay", false))
                check(evaluate(test, inputs).raw_data() == output.raw_data(), "Random seed replay differs");
            if (!test.contains("upstream")) continue;
            const Json& seed = test.at("upstream");
            const Tensor upstream = Tensor::from_raw(shape(seed), DType(seed.at("dtype").integer()), bytes(seed), backend);
            output.backward(upstream);
            const elman::Json::Array& gradients = test.at("gradients").array();
            check(gradients.size() == inputs.size(), "Reference gradient count differs");
            for (size_t i = 0; i < inputs.size(); ++i) {
                if (gradients[i].is_null()) continue;
                check(bool(inputs[i].node()->gradient), "Reference gradient was not produced");
                const elman::Json expected = observed_mask ? specification(conditioned_dropout(upstream, scales.second)) : gradients[i];
                compare(Tensor(inputs[i].node()->gradient), expected, name + "/gradient/" + std::to_string(i));
            }
        } catch (const std::exception& error) {
            ++failed;
            std::cerr << test.at("name").string() << ": " << error.what() << '\n';
        }
    }
    if (failed) throw std::runtime_error(std::to_string(failed) + " low precision reference cases failed");
}

template<class Callback>
void run_checked(Backend backend, const std::string& name, Callback&& callback) {
    const size_t live = memory_stats(backend).live_bytes();
    const int previous_checks = checks;
    const size_t previous_elements = elements;
    const size_t previous_nan_encodings = nan_encodings;
    try {
        {
            // Release the suite's lazy BLAS workspace before checking storage lifetime.
            ExecutionStream execution(backend);
            StreamScope execution_scope(execution, false);
            callback();
            execution.synchronize();
        }
        check(memory_stats(backend).live_bytes() == live, "Execution leaked storage");
    } catch (const std::exception& error) {
        throw std::runtime_error(name + ": " + error.what());
    }
    std::cout << name << ": " << checks - previous_checks << " checks, "
              << elements - previous_elements << " reference elements, "
              << nan_encodings - previous_nan_encodings << " NaN representation differences; 0 failed\n";
}

void run_suite(Backend backend, const Suite& suite) {
    run_checked(backend, std::string(suite.option + 2), [&] { suite.run(backend); });
}

}  // namespace kernals

namespace optimizer_reference {
using namespace elman;
size_t checks=0,elements=0,nan_encodings=0,steps=0;
void check(bool valid,const std::string& message){++checks;if(!valid)throw std::runtime_error(message);}
Shape shape(const Json& value){Shape result;for(const elman::Json& d:value.at("shape").array())result.push_back(d.integer());return result;}
std::vector<uint8_t> bytes(const Json& value){
    const std::string& hex=value.at("raw_hex").string();std::vector<uint8_t> result;
    int (*const digit)(char)=[](char c){if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::runtime_error("Invalid optimizer reference hex");};
    check(hex.size()%2==0,"Odd optimizer reference hex length");
    for(size_t i=0;i<hex.size();i+=2)result.push_back(uint8_t(16*digit(hex[i])+digit(hex[i+1])));return result;
}
Tensor tensor(const Json& value,Backend backend){return Tensor::from_raw(shape(value),DType(value.at("dtype").integer()),bytes(value),backend);}
bool nan_code(uint32_t code,DType dtype){
    if(dtype==DType::Float32)return (code&0x7fffffffu)>0x7f800000u;
    if(dtype==DType::Float8E4M3FN)return (code&127)==127;
    if(dtype==DType::Float8E4M3FNUZ||dtype==DType::Float8E5M2FNUZ)return code==128;
    if(dtype==DType::Float8E5M2)return (code&127)>124;
    return dtype==DType::Float8E8M0FNU&&code==255;
}
void compare(const Tensor& actual,const Json& expected,const std::string& context){
    check(actual.dtype()==DType(expected.at("dtype").integer())&&actual.shape()==shape(expected),context+": layout differs");
    const std::vector<uint8_t> got=actual.raw_data(),wanted=bytes(expected);check(got.size()==wanted.size(),context+": storage length differs");elements+=actual.numel();
    if(actual.dtype()==DType::Float4E2M1FN){check(got==wanted,context+": packed FP4 bytes differ");return;}
    const size_t width=dtype_info(actual.dtype()).bits/8;
    for(size_t i=0;i<actual.numel();++i){uint32_t a=0,b=0;std::memcpy(&a,got.data()+i*width,width);std::memcpy(&b,wanted.data()+i*width,width);
        if(a==b)continue;if(nan_code(a,actual.dtype())&&nan_code(b,actual.dtype())){++nan_encodings;continue;}
        throw std::runtime_error(context+": exact bits differ at "+std::to_string(i)+" (actual "+std::to_string(a)+", expected "+std::to_string(b)+")");}
    ++checks;
}
double scalar(const Json& words){const elman::Json::Array& a=words.array();uint64_t bits=uint64_t(uint32_t(a[0].integer()))|(uint64_t(uint32_t(a[1].integer()))<<32);double value;std::memcpy(&value,&bits,8);return value;}
std::vector<int> attributes(const Json& test,const Json& record,size_t index){
    const Json& settings=test.at("settings");const std::string algorithm=test.at("algorithm").string();
    const uint64_t step=record.contains("step")?std::stoull(record.at("step").string()):index+1;
    const double beta1=settings.at("beta1").number(),beta2=settings.at("beta2").number();
    const double parameters[]={record.at("rate").number(),record.at("gradient_scale").number(),settings.at("weight_decay").number(),
        settings.at("momentum").number(),beta1,beta2,settings.at("epsilon").number(),
        record.contains("correction1_bits")?scalar(record.at("correction1_bits")):1.-std::pow(beta1,double(step)),
        record.contains("correction2_sqrt_bits")?scalar(record.at("correction2_sqrt_bits")):std::pow(1.-std::pow(beta2,double(step)),.5)};
    std::vector<int> result(22);result[0]=algorithm=="sgd"?0:algorithm=="adam"?1:2;
    result[2]=record.contains("first_step")?int(record.at("first_step").boolean()):step==1?1:0;
    for(int i=0;i<9;++i)std::memcpy(result.data()+4+2*i,&parameters[i],8);return result;
}
void raw_case(const Json& test,Backend backend){
    size_t index=0;
    for(const elman::Json& record:test.at("steps").array()){
        std::vector<Tensor> inputs{tensor(record.at("parameter_before"),backend),tensor(record.at("gradient"),backend),
            tensor(record.at("first_moment_before"),backend),tensor(record.at("second_moment_before"),backend)};
        std::vector<int> params=attributes(test,record,index);const char* fields[]={"parameter","first_moment","second_moment","parameter_float32"};
        for(int mode=0;mode<4;++mode){params[1]=mode;const elman::DType dtype=mode==0?inputs[0].dtype():DType::Float32;
            Tensor output=dispatch_tensor("optimizer_step",inputs[0].shape(),inputs,params,dtype);
            compare(output,record.at(fields[mode]),test.at("name").string()+"/step/"+std::to_string(index)+"/"+fields[mode]);}
        ++index;++steps;
    }
}
class Probe final:public Model {
public:
    Probe(const Json& test,Backend backend):Model("LowOptimizerProbe","optimizer-probe",backend,1){
        size_t index=0;for(const elman::Json& value:test.at("parameters").array())
            parameters_.push_back({"parameter"+std::to_string(index++),Tensor::from_raw(shape(value),DType(value.at("dtype").integer()),bytes(value),backend,true)});
    }
    Tensor forward(const Tensor& input,bool,uint32_t) override{return input;}
    Shape sample_shape() const override{return {1};}
    int output_classes() const override{return 1;}
};
using Snapshot=std::vector<std::vector<uint8_t>>;
Snapshot snapshot(const Model& model,const TrainingState& state){
    Snapshot result;std::vector<uint8_t> step(sizeof(state.step));std::memcpy(step.data(),&state.step,step.size());result.push_back(step);
    result.emplace_back(state.optimizer.begin(),state.optimizer.end());result.push_back(state.sgd_buffer_initialized);
    for(const elman::Parameter& parameter:model.parameters())result.push_back(parameter.value.raw_data());
    for(const std::vector<elman::Tensor>* group:{&state.first_moment_float,&state.second_moment_float})
        for(const elman::Tensor& moment:*group)result.push_back(moment.defined()?moment.raw_data():std::vector<uint8_t>{});
    return result;
}
void frontend_expected(const Probe& model,const TrainingState& state,const Json& record,bool initial,const Json& test,size_t case_steps){
    check(state.step==std::stoull(test.at("initial_step").string())+case_steps,"Optimizer frontend step metadata differs");
    check(state.optimizer==test.at("algorithm").string(),"Optimizer frontend algorithm differs");
    for(size_t i=0;i<model.parameters().size();++i){
        const Tensor& value=model.parameters()[i].value;const std::string context=test.at("name").string()+"/parameter/"+std::to_string(i);
        compare(value,record.at("parameters").at(i),context);
        if(!initial)compare(cast(value.detach(),DType::Float32),record.at("parameter_decoded").at(i),context+"/decoded");
        const elman::Tensor& first=state.first_moment_float[i];const elman::Tensor& second=state.second_moment_float[i];
        check(first.backend()==value.backend()&&second.backend()==value.backend()&&!first.requires_grad()&&!second.requires_grad(),context+": moment descriptor differs");
        compare(first,record.at(initial?"first_moment_initial":"first_moment").at(i),context+"/first");
        compare(second,record.at(initial?"second_moment_initial":"second_moment").at(i),context+"/second");
        check(state.first_moment[i].empty()&&state.second_moment[i].empty()&&!state.first_moment_double[i].defined()&&!state.second_moment_double[i].defined(),context+": unexpected legacy moments");
        const bool history=test.at("algorithm").string()=="sgd"&&(!initial||(test.contains("sgd_history_initial")?test.at("sgd_history_initial").at(i).boolean():std::stoull(test.at("initial_step").string())>0));
        check(state.sgd_buffer_initialized[i]==int(history),context+": SGD history differs");
    }
}
void apply_step(Probe& model,Optimizer& optimizer,const Json& record){
    std::vector<uint64_t> versions;
    for(size_t i=0;i<model.parameters().size();++i){Tensor& value=model.parameters()[i].value;
        versions.push_back(value.node()->value->version);value.zero_grad();value.backward(tensor(record.at("gradients").at(i),value.backend()));}
    const TrainingState old=optimizer.state();Snapshot moments;
    for(const std::vector<elman::Tensor>* group:{&old.first_moment_float,&old.second_moment_float})for(const elman::Tensor& moment:*group)moments.push_back(moment.raw_data());
    const double norm=optimizer.update(model,record.at("rate").number()),expected=scalar(record.at("norm_bits"));
    check(std::memcmp(&norm,&expected,sizeof(norm))==0,"Optimizer frontend norm bits differ");
    size_t index=0;for(const std::vector<elman::Tensor>* group:{&old.first_moment_float,&old.second_moment_float})for(const elman::Tensor& moment:*group)
        check(moment.raw_data()==moments[index++],"Optimizer mutated a retained state snapshot");
    for(size_t i=0;i<versions.size();++i)check(model.parameters()[i].value.node()->value->version==versions[i]+1,"Optimizer parameter version did not advance exactly once");
}
void frontend_case(const Json& test,Backend backend,const std::filesystem::path& directory,size_t case_index){
    Probe model(test,backend);elman::Json::Object settings=test.at("settings").object();settings["optimizer"]=test.at("algorithm");
    Optimizer optimizer(Json(settings),model);TrainingState& state=optimizer.state();state.step=std::stoull(test.at("initial_step").string());
    for(size_t i=0;i<model.parameters().size();++i){
        state.first_moment_float[i]=tensor(test.at("first_moment_initial").at(i),backend);
        state.second_moment_float[i]=tensor(test.at("second_moment_initial").at(i),backend);
        state.sgd_buffer_initialized[i]=test.at("algorithm").string()=="sgd"&&(test.contains("sgd_history_initial")?test.at("sgd_history_initial").at(i).boolean():state.step>0);
    }
    const Json definition=Json::Object{{"architecture","optimizer-probe"}};
    std::unique_ptr<Probe> restored;std::unique_ptr<Optimizer> resumed;
    LocalFunction checkpoint{[&](const std::string& suffix){
        std::filesystem::path path=directory/("frontend-"+std::to_string(case_index)+suffix+".bin");save_checkpoint(path.string(),model,definition,optimizer.state());
        restored=std::make_unique<Probe>(test,backend);resumed=std::make_unique<Optimizer>(Json(settings),*restored);
        load_checkpoint(path.string(),*restored,definition,&resumed->state());
        check(snapshot(model,optimizer.state())==snapshot(*restored,resumed->state()),"Checkpoint changed low-format parameters, moments or history");
    }};
    frontend_expected(model,optimizer.state(),test,true,test,0);
    if(test.bool_or("checkpoint_initial",false)){checkpoint("-initial");frontend_expected(*restored,resumed->state(),test,true,test,0);}
    size_t index=0;for(const elman::Json& record:test.at("steps").array()){
        apply_step(model,optimizer,record);++steps;frontend_expected(model,optimizer.state(),record,false,test,index+1);
        if(resumed){apply_step(*restored,*resumed,record);frontend_expected(*restored,resumed->state(),record,false,test,index+1);
            check(snapshot(model,optimizer.state())==snapshot(*restored,resumed->state()),"Resumed optimizer trajectory differs");}
        if(index++==size_t(test.at("checkpoint_after").integer()))checkpoint("");
    }
}

int run(Backend backend, const std::filesystem::path& fixture, bool frontend,
        const std::filesystem::path& checkpoint_directory) {
    if (frontend) std::filesystem::create_directories(checkpoint_directory);
    const Json suite = Json::read(fixture);
    check(suite.at("backend").string() == backend_name(backend), "Optimizer reference backend differs");
    const size_t live = memory_stats(backend).live_bytes();
    size_t cases = 0, failed = 0;
    {
        AllocationPool pool(backend);
        ExecutionStream stream(backend);
        StreamScope scope(stream, false);
        if (frontend) {
            for (const elman::Json& test : suite.at("frontend_cases").array()) {
                const size_t previous_steps = steps;
                try {
                    frontend_case(test, backend, checkpoint_directory, cases);
                } catch (const std::exception& error) {
                    steps = previous_steps;
                    ++failed;
                    std::cerr << test.at("name").string() << ": " << error.what() << '\n';
                }
                ++cases;
            }
        } else {
            for (const char* group : {"cases", "extensions"}) {
                for (const elman::Json& test : suite.at(group).array()) {
                    ++cases;
                    try {
                        raw_case(test, backend);
                    } catch (const std::exception& error) {
                        ++failed;
                        std::cerr << test.at("name").string() << ": " << error.what() << '\n';
                    }
                }
            }
        }
        stream.synchronize();
    }
    check(memory_stats(backend).live_bytes() == live, "Optimizer reference leaked storage");
    std::cout << "Low precision optimizer: " << cases << " cases, " << steps << " steps, "
              << checks << " checks, " << elements << " reference elements, " << nan_encodings
              << " NaN representation differences; " << failed << " failed\n";
    return failed ? 1 : 0;
}

}  // namespace optimizer_reference

namespace optimizer_guards {
using namespace elman;
size_t checks=0;
void check(bool value,const char* message){++checks;if(!value)throw std::runtime_error(message);}
template<class F> void rejects(F action){bool rejected=false;try{action();}catch(const std::exception&){rejected=true;}check(rejected,"Invalid low-format optimizer operation was accepted");}
const DType low_types[]={DType::Float8E4M3FN,DType::Float8E4M3FNUZ,DType::Float8E5M2,DType::Float8E5M2FNUZ,DType::Float8E8M0FNU,DType::Float4E2M1FN};
Json settings(const std::string& algorithm){return Json::Object{{"optimizer",algorithm},{"learning_rate",.01},{"momentum",.6},{"weight_decay",.01},{"clip_grad_norm",0.}};}
class Probe final:public Model {
public:
    Probe(Backend backend,bool mixed=true):Model("MixedLowOptimizerProbe","optimizer-guards",backend,1){
        for(DType dtype:low_types)parameters_.push_back({"low"+std::to_string(int(dtype)),Tensor::from_data({3},{1.,.5,2.},backend,true,dtype)});
        parameters_.push_back({"scalar",Tensor::from_data({}, {-0.f},backend,true,DType::Float4E2M1FN)});
        parameters_.push_back({"empty",Tensor::zeros({2,0,3},backend,true,DType::Float8E5M2)});
        if(mixed){parameters_.push_back({"double",Tensor::from_double_data({2},{1.+std::ldexp(1.,-40),1e-100},backend,true)});
            for(elman::DType dtype:{DType::Float32,DType::Float16,DType::BFloat16})parameters_.push_back({"native"+std::to_string(int(dtype)),Tensor::from_data({3},{1.f,.5f,2.f},backend,true,dtype)});}
    }
    Tensor forward(const Tensor& input,bool,uint32_t) override{return input;}
    Shape sample_shape() const override{return {1};}
    int output_classes() const override{return 1;}
};
using Snapshot=std::vector<std::vector<uint8_t>>;
template<class T> std::vector<uint8_t> raw(const std::vector<T>& values){std::vector<uint8_t> result(values.size()*sizeof(T));if(!result.empty())std::memcpy(result.data(),values.data(),result.size());return result;}
Snapshot snapshot(const Model& model,const TrainingState& state){
    Snapshot result{raw(std::vector<uint64_t>{state.step}),std::vector<uint8_t>(state.optimizer.begin(),state.optimizer.end()),state.sgd_buffer_initialized};
    for(const elman::Parameter& parameter:model.parameters())result.push_back(parameter.value.raw_data());
    for(const std::vector<std::vector<float>>* group:{&state.first_moment,&state.second_moment}){result.push_back(raw(std::vector<size_t>{group->size()}));for(const std::vector<float>& row:*group)result.push_back(raw(row));}
    for(const std::vector<elman::Tensor>* group:{&state.first_moment_float,&state.second_moment_float,&state.first_moment_double,&state.second_moment_double}){
        result.push_back(raw(std::vector<size_t>{group->size()}));
        for(const elman::Tensor& row:*group){result.push_back(raw(std::vector<int>{row.defined()?int(row.dtype()):-1}));result.push_back(row.defined()?row.raw_data():std::vector<uint8_t>{});}}
    return result;
}
std::vector<uint64_t> versions(const Model& model){std::vector<uint64_t> result;for(const elman::Parameter& p:model.parameters())result.push_back(p.value.node()->value->version);return result;}
void gradients(Probe& model,float value=1.f){for(elman::Parameter& parameter:model.parameters()){
    Tensor& x=parameter.value;x.zero_grad();
    if(x.dtype()==DType::Float64)x.backward(Tensor::from_double_data(x.shape(),std::vector<double>(x.numel(),double(value)),x.backend()));
    else x.backward(Tensor::from_data(x.shape(),std::vector<float>(x.numel(),value),x.backend(),false,x.dtype()));}}
template<class F>void atomic_reject(Probe& model,Optimizer& optimizer,F action){const Snapshot before=snapshot(model,optimizer.state());const std::vector<uint64_t> stamps=versions(model);
    rejects(action);check(snapshot(model,optimizer.state())==before&&versions(model)==stamps,"Rejected optimizer action partially mutated parameters, versions or state");}
std::vector<int> attributes(){std::vector<int> result(22);result[2]=1;const double values[]={.01,1.,.01,.6,.9,.999,1e-8,.1,std::sqrt(.001)};
    for(int i=0;i<9;++i)std::memcpy(result.data()+4+2*i,values+i,8);return result;}
KernelParams descriptor(const Tensor& output,const std::vector<Tensor>& inputs,const std::vector<int>& attrs){
    KernelParams p;p.v[0]=int(output.numel());p.v[1]=int(output.shape().size());p.v[255]=int(output.dtype());
    for(size_t d=0;d<output.shape().size();++d)p.v[2+d]=output.shape()[d];
    for(size_t i=0;i<inputs.size();++i){const int slot=KernelParams::input_slot(KernelInput(i));p.v[slot]=int(inputs[i].shape().size());p.v[KernelParams::input_dtype_slot(KernelInput(i))]=int(inputs[i].dtype());
        int stride=1;for(int d=int(inputs[i].shape().size())-1;d>=0;--d){p.v[slot+1+d]=inputs[i].shape()[d];p.v[slot+9+d]=stride;stride*=inputs[i].shape()[d];}}
    for(size_t i=0;i<attrs.size();++i)p.v[64+i]=attrs[i];return p;
}
void raw_validation(Backend backend){for(DType dtype:low_types){
    std::vector<Tensor> inputs{Tensor::ones({2,3},backend,false,dtype),Tensor::ones({2,3},backend,false,dtype),Tensor::zeros({2,3},backend),Tensor::zeros({2,3},backend)};
    Tensor output=Tensor::zeros({2,3},backend,false,dtype);KernelParams p=descriptor(output,inputs,attributes());std::vector<BufferPtr> buffers;for(const elman::Tensor& x:inputs)buffers.push_back(x.node()->value);
    LocalFunction dispatch{[&](const KernelParams& candidate){backend_dispatch("optimizer_step",output.node()->value,buffers,candidate);}};dispatch(p);const std::vector<uint8_t> unchanged=output.raw_data();
    for(int slot:{0,1,2,255}){KernelParams bad=p;bad.v[slot]=slot==1?9:-1;rejects([&]{dispatch(bad);});}
    for(int i=0;i<4;++i)for(int offset:{0,1,9}){KernelParams bad=p;bad.v[KernelParams::input_slot(KernelInput(i))+offset]=-1;rejects([&]{dispatch(bad);});}
    for(int i=0;i<4;++i){KernelParams bad=p;bad.v[KernelParams::input_dtype_slot(KernelInput(i))]=int(DType::Int32);rejects([&]{dispatch(bad);});
        std::vector<BufferPtr> missing=buffers;missing[i].reset();rejects([&]{backend_dispatch("optimizer_step",output.node()->value,missing,p);});}
    for(int count:{0,1,2,3}){std::vector<BufferPtr> missing=buffers;missing.resize(count);rejects([&]{backend_dispatch("optimizer_step",output.node()->value,missing,p);});}
    for(int slot:{0,1,2,3,22,63}){KernelParams bad=p;bad.v[64+slot]=7;rejects([&]{dispatch(bad);});}
    for(int scalar=0;scalar<9;++scalar)for(double value:{double(NAN),double(INFINITY),-1.}){KernelParams bad=p;std::memcpy(bad.v+68+2*scalar,&value,8);rejects([&]{dispatch(bad);});}
    for(int scalar:{3,4,5,7,8}){KernelParams bad=p;const double value=2.;std::memcpy(bad.v+68+2*scalar,&value,8);rejects([&]{dispatch(bad);});}
    for(int scalar:{6,7,8}){KernelParams bad=p;const double value=0.;std::memcpy(bad.v+68+2*scalar,&value,8);rejects([&]{dispatch(bad);});}
    for(int mode=0;mode<4;++mode){std::vector<int> attrs=attributes();attrs[1]=mode;const elman::Tensor& alias=inputs[mode==0?0:2];KernelParams alias_p=descriptor(alias,inputs,attrs);
        rejects([&]{backend_dispatch("optimizer_step",alias.node()->value,buffers,alias_p);});}
    Tensor owner=Tensor::ones({10},backend,false,dtype),a=contiguous_view(owner,{2,3},0),overlap=contiguous_view(owner,{2,3},2);std::vector<Tensor> partial_inputs=inputs;partial_inputs[0]=a;
    KernelParams partial=descriptor(overlap,partial_inputs,attributes());std::vector<BufferPtr> partial_buffers=buffers;partial_buffers[0]=a.node()->value;
    rejects([&]{backend_dispatch("optimizer_step",overlap.node()->value,partial_buffers,partial);});
    std::vector<Tensor> native=inputs;native[0]=Tensor::ones({2,3},backend);native[1]=native[0];
    rejects([&]{dispatch_tensor("optimizer_step",{2,3},native,attributes(),DType::Float32);});
    std::vector<Tensor> mixed=inputs;mixed[1]=Tensor::ones({2,3},backend,false,dtype==low_types[0]?low_types[1]:low_types[0]);
    rejects([&]{dispatch_tensor("optimizer_step",{2,3},mixed,attributes(),dtype);});
    KernelParams overflow=p;overflow.v[2]=INT32_MAX;overflow.v[3]=2;
    for(int i=0;i<4;++i){overflow.v[KernelParams::input_slot(KernelInput(i))+1]=INT32_MAX;overflow.v[KernelParams::input_slot(KernelInput(i))+2]=2;}
    rejects([&]{dispatch(overflow);});
    check(output.raw_data()==unchanged,"Rejected raw optimizer descriptor modified output bytes");
}}
void validation(Backend backend){for(const std::string algorithm:{"sgd","adam","adamw"}){
    Probe model(backend);Optimizer optimizer(settings(algorithm),model);gradients(model);
    for(double rate:{-1.,double(NAN),double(INFINITY)})atomic_reject(model,optimizer,[&]{optimizer.update(model,rate);});
    for(int kind=0;kind<23;++kind){gradients(model);const TrainingState saved=optimizer.state();TrainingState& state=optimizer.state();Tensor& value=model.parameters()[5].value;const std::vector<uint8_t> parameter=value.raw_data();
        if(kind==0)state.first_moment_float.pop_back();
        if(kind==1)state.second_moment_float.pop_back();
        if(kind==2)state.sgd_buffer_initialized.pop_back();
        if(kind==3)state.sgd_buffer_initialized[5]=2;
        if(kind==4)state.first_moment_float[5]=Tensor::zeros({2},backend);
        if(kind==5)state.second_moment_float[5]=Tensor::zeros({3},backend,false,DType::Float64);
        if(kind==6)state.first_moment_float[5]=Tensor::ones({3},backend,true);
        if(kind==7)state.first_moment_float[5]=Tensor::from_data({3},{NAN,0,0},backend);
        if(kind==8)state.second_moment_float[5]=Tensor::from_data({3},{-1,0,0},backend);
        if(kind==9)state.second_moment_float[5]=Tensor::from_data({3},{INFINITY,0,0},backend);
        if(kind==10)state.first_moment[5]={0};
        if(kind==11)state.first_moment_double[5]=Tensor::zeros({3},backend,false,DType::Float64);
        if(kind==12)state.step=uint64_t(std::numeric_limits<int64_t>::max());
        if(kind==13)state.optimizer="other";
        if(kind==14)value.zero_grad();
        if(kind==15){Tensor& bad=model.parameters()[0].value;bad.zero_grad();bad.backward(Tensor::from_data({3},{NAN,1,1},backend,false,bad.dtype()));}
        if(kind==16){Tensor& bad=model.parameters()[0].value;bad.assign({NAN,1,1});}
        if(kind==17)state.first_moment_float[8]=Tensor::zeros({2},backend);
        if(kind==18)state.sgd_buffer_initialized[8]=1;
        if(kind==19)state.second_moment_double[5]=Tensor::zeros({3},backend,false,DType::Float64);
        if(kind==20)state.second_moment[5]={0};
        if(kind==21)state.first_moment_float[5]=Tensor{};
        if(kind==22)state.second_moment_float[5]=Tensor{};
        atomic_reject(model,optimizer,[&]{optimizer.update(model,.01);});
        state=saved;value.assign_raw(parameter);model.parameters()[0].value.assign({1,.5,2});
    }
    gradients(model);atomic_reject(model,optimizer,[&]{optimizer.update(model,1e300);});
    if(backend==Backend::CUDA){const TrainingState saved=optimizer.state();optimizer.state().first_moment_float[5]=Tensor::zeros({3},Backend::CPU);
        atomic_reject(model,optimizer,[&]{optimizer.update(model,.01);});optimizer.state()=saved;}
    // Packed saturation must not conceal a nonfinite pre-quantization result.
    Probe packed(backend,false);packed.parameters().erase(packed.parameters().begin(),packed.parameters().begin()+5);
    elman::Json::Object config=settings(algorithm).object();config["weight_decay"]=1e300;Optimizer overflow(Json(config),packed);gradients(packed);
    atomic_reject(packed,overflow,[&]{overflow.update(packed,.01);});
    gradients(model);Tensor graph=mul(model.parameters()[0].value,model.parameters()[0].value);optimizer.update(model,.01);rejects([&]{graph.backward();});
    check(model.parameters()[0].value.requires_grad(),"Optimizer detached a low-format parameter leaf");
}}
std::vector<uint8_t> read_bytes(const std::filesystem::path& path){std::ifstream file(path,std::ios::binary);return {std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>()};}
void write_bytes(const std::filesystem::path& path,const std::vector<uint8_t>& data){std::ofstream file(path,std::ios::binary);file.write(reinterpret_cast<const char*>(data.data()),std::streamsize(data.size()));check(bool(file),"Cannot write low optimizer fixture");}
void checksum(std::vector<uint8_t>& data){uint64_t hash=14695981039346656037ull;for(size_t i=0;i<data.size()-8;++i){hash^=data[i];hash*=1099511628211ull;}for(int i=0;i<8;++i)data[data.size()-8+i]=uint8_t(hash>>(8*i));}
struct Offsets{size_t dtype,length,parameter,history,first_length,first,second_length,second;};
std::vector<Offsets> offsets(const std::vector<uint8_t>& data){size_t cursor=12;
    LocalFunction integer{[&](int width){uint64_t result=0;for(int i=0;i<width;++i)result|=uint64_t(data.at(cursor++))<<(8*i);return result;}};
    LocalFunction string{[&]{uint64_t length=integer(4);cursor+=size_t(length);}};string();string();integer(8);string();const uint64_t count=integer(4);std::vector<Offsets> result;
    for(uint64_t i=0;i<count;++i){string();uint64_t rank=integer(4);cursor+=4*size_t(rank);Offsets o{};o.dtype=cursor;DType dtype=DType(integer(4));o.length=cursor;uint64_t length=integer(8);o.parameter=cursor;cursor+=size_t(length);
        if(is_low_precision_float(dtype)){o.history=cursor;integer(4);}o.first_length=cursor;length=integer(8);o.first=cursor;cursor+=size_t(length)*(dtype==DType::Float64?8:4);
        o.second_length=cursor;length=integer(8);o.second=cursor;cursor+=size_t(length)*(dtype==DType::Float64?8:4);result.push_back(o);}
    check(cursor==data.size()-8,"Invalid version 3 checkpoint layout");return result;
}
void checkpoint(Backend backend,const std::filesystem::path& directory){const Json definition=Json::Object{{"architecture","optimizer-guards"}};
    for(const std::string algorithm:{"sgd","adam","adamw"}){
        Probe model(backend),restored(backend);Optimizer optimizer(settings(algorithm),model),resumed(settings(algorithm),restored);
        // Preserve the unused nibble too: raw storage is the v3 persistence contract.
        std::vector<uint8_t> packed=model.parameters()[5].value.raw_data();packed.back()|=0xf0;model.parameters()[5].value.assign_raw(packed);
        const std::filesystem::path path=directory/(algorithm+".bin"),bad=directory/(algorithm+"-bad.bin");
        save_checkpoint(path.string(),model,definition,optimizer.state());std::vector<uint8_t> data=read_bytes(path);check(data.at(8)==3,"Low-format checkpoint did not use version 3");
        const std::vector<uint64_t> stamps=versions(restored);Tensor stale=mul(restored.parameters()[0].value,restored.parameters()[0].value);
        load_checkpoint(path.string(),restored,definition,&resumed.state());check(snapshot(model,optimizer.state())==snapshot(restored,resumed.state()),"Mixed initial checkpoint lost raw parameters or typed moments");
        const std::vector<uint64_t> after=versions(restored);for(size_t i=0;i<stamps.size();++i)check(after[i]==stamps[i]+1,"Checkpoint restore did not advance parameter versions exactly once");
        rejects([&]{stale.backward();});
        for(int i=0;i<5;++i){gradients(model);gradients(restored);optimizer.update(model,.01);resumed.update(restored,.01);check(snapshot(model,optimizer.state())==snapshot(restored,resumed.state()),"Mixed resumed optimizer differs");}
        save_checkpoint(path.string(),model,definition,optimizer.state());data=read_bytes(path);const std::vector<Offsets> fields=offsets(data);
        if(backend==Backend::CUDA){Probe cpu(Backend::CPU);Optimizer cpu_optimizer(settings(algorithm),cpu);load_checkpoint(path.string(),cpu,definition,&cpu_optimizer.state());
            check(snapshot(cpu,cpu_optimizer.state())==snapshot(model,optimizer.state()),"Version 3 cross-backend restore differs");std::filesystem::path cross=directory/(algorithm+"-cpu.bin");save_checkpoint(cross.string(),cpu,definition,cpu_optimizer.state());check(read_bytes(cross)==data,"Cross-backend version 3 bytes differ");}
        LocalFunction reject_file{[&](std::vector<uint8_t> invalid,bool rehash=true){if(rehash)checksum(invalid);write_bytes(bad,invalid);atomic_reject(restored,resumed,[&]{load_checkpoint(bad.string(),restored,definition,&resumed.state());});}};
        for(size_t cut:{size_t(0),size_t(1),size_t(8),size_t(12),data.size()/2,data.size()-1})reject_file(std::vector<uint8_t>(data.begin(),data.begin()+cut),false);
        for(size_t location:{size_t(0),size_t(8),data.size()-1}){std::vector<uint8_t> invalid=data;invalid[location]^=0x40;reject_file(invalid,false);}
        std::vector<uint8_t> trailing=data;trailing.push_back(0);reject_file(trailing,false);
        for(size_t i=0;i<fields.size();++i){for(size_t location:{fields[i].dtype,fields[i].length,fields[i].first_length,fields[i].second_length}){std::vector<uint8_t> invalid=data;invalid[location]=255;reject_file(invalid);}
            if(fields[i].history){std::vector<uint8_t> invalid=data;invalid[fields[i].history]=2;reject_file(invalid);
                if(algorithm!="sgd"){invalid=data;invalid[fields[i].history]=1;reject_file(invalid);}}
            if(i<6&&i!=5){std::vector<uint8_t> invalid=data;invalid[fields[i].parameter]=i==0?127:i==2?127:i==4?255:128;reject_file(invalid);}
            if(i<6){for(size_t location:{fields[i].first,fields[i].second}){std::vector<uint8_t> invalid=data;const float value=location==fields[i].first?float(NAN):-1.f;std::memcpy(invalid.data()+location,&value,4);reject_file(invalid);}}
        }
        for(int kind=0;kind<8;++kind){TrainingState invalid=optimizer.state();
            if(kind==0)invalid.first_moment_float[0]=Tensor::zeros({2},backend);
            if(kind==1)invalid.second_moment_float[0]=Tensor::from_data({3},{-1,0,0},backend);
            if(kind==2)invalid.sgd_buffer_initialized[0]=2;
            if(kind==3)invalid.first_moment_float[8]=Tensor::zeros({2},backend);
            if(kind==4)invalid.first_moment[0]={0};
            if(kind==5)invalid.first_moment_double[0]=Tensor::zeros({3},backend,false,DType::Float64);
            if(kind==6)invalid.first_moment_float[0]=Tensor::from_data({3},{NAN,0,0},backend);
            if(kind==7)invalid.second_moment_float.pop_back();
            rejects([&]{save_checkpoint(path.string(),model,definition,invalid);});check(read_bytes(path)==data&&!std::filesystem::exists(path.string()+".tmp"),"Invalid save damaged checkpoint or left temporary output");}
        if(backend==Backend::CUDA){TrainingState invalid=optimizer.state();invalid.second_moment_float[5]=Tensor::zeros({3},Backend::CPU);
            rejects([&]{save_checkpoint(path.string(),model,definition,invalid);});check(read_bytes(path)==data,"Wrong-backend moment damaged a valid checkpoint");}
    }
}
void execution(Backend backend){for(bool pooled:{false,true})for(bool asynchronous:{false,true}){
    AllocationPool pool(backend,0,pooled);ExecutionStream producer(backend),consumer(backend);
    for(int i=0;i<3;++i){Probe model(backend);Optimizer optimizer(settings("adamw"),model);
        {StreamScope scope(producer,asynchronous,true);gradients(model);optimizer.update(model,.01);}
        {StreamScope scope(consumer,asynchronous,true);gradients(model);optimizer.update(model,.01);}
        check(optimizer.state().step==2,"Asynchronous mixed optimizer step differs");}
    producer.synchronize();consumer.synchronize();if(pooled)check(pool.stats().reuses>0,"Low optimizer did not exercise pool reuse");
}}

void run(elman::Backend backend,const std::filesystem::path& directory){
    check(!std::filesystem::exists(directory),"Low optimizer guard directory already exists");std::filesystem::create_directories(directory);const size_t before=elman::memory_stats(backend).live_bytes();
    raw_validation(backend);validation(backend);checkpoint(backend,directory);execution(backend);
    check(elman::memory_stats(backend).live_bytes()==before,"Low optimizer guards leaked storage");std::cout<<"Low precision optimizer guards: "<<checks<<" passed, 0 failed\n";
}

}  // namespace optimizer_guards

void run_float8(elman::Backend backend, const std::filesystem::path& output) {
    for (const kernals::Suite& suite : kernals::float8_suites) kernals::run_suite(backend, suite);
    const std::chrono::steady_clock::rep stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    optimizer_guards::run(backend, output / std::to_string(stamp));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5)
            throw std::invalid_argument("Usage: elman_float8_tests <cpu|cuda> [--suite OUTPUT|--float16|suite flag|fixture]");
        const elman::Backend backend = elman::parse_backend(argv[1]);
        if (!elman::backend_available(backend)) return 77;
        if (argc == 2) {
            run_float8(backend, std::filesystem::path("TestArtifacts/Float8") / elman::backend_name(backend));
        } else if (argc == 4 && std::string(argv[2]) == "--suite") {
            run_float8(backend, argv[3]);
        } else if (argc == 4 && std::string(argv[2]) == "--guards") {
            optimizer_guards::run(backend, argv[3]);
        } else if (argc == 4 && std::string(argv[3]) == "--raw") {
            return optimizer_reference::run(backend, argv[2], false, {});
        } else if (argc == 5 && std::string(argv[3]) == "--frontend") {
            return optimizer_reference::run(backend, argv[2], true, argv[4]);
        } else if (argc == 3) {
            const std::string option = argv[2];
            for (const kernals::Suite& suite : kernals::float8_suites) {
                if (option != suite.option) continue;
                kernals::run_suite(backend, suite);
                return 0;
            }
            for (const kernals::Suite& suite : kernals::other_suites) {
                if (option != suite.option) continue;
                kernals::run_suite(backend, suite);
                return 0;
            }
            kernals::run_checked(backend, option, [&] { kernals::reference(backend, option); });
        } else {
            throw std::invalid_argument("Invalid Float8 test mode");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
