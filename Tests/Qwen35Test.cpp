#include "Models/LLM/Qwen35.h"
#include "Models/LLM/DecoderFactory.h"
#include "Inference/LLM/Generation.h"
#include "IO/Json.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void integer(std::ostream& out, uint64_t value, int bytes) { for (int i = 0; i < bytes; ++i) out.put(char(value >> (i * 8))); }
void string(std::ostream& out, const std::string& value) { integer(out, value.size(), 8); out.write(value.data(), value.size()); }
void fixture(const std::filesystem::path& path) {
    std::vector<std::pair<std::string, Tensor>> tensors;
    auto add = [&](const std::string& name, const Shape& shape, bool norm = false, bool decay = false) {
        size_t count = 1; for (int dim : shape) count *= dim;
        std::vector<float> values(count);
        for (size_t i = 0; i < count; ++i) values[i] = decay ? -.5f : norm ? 1.f : float(.1 * std::sin(i * .27 + tensors.size()));
        tensors.emplace_back(name, Tensor::from_data(shape, values));
    };
    add("token_embd.weight", {16,8}); add("output_norm.weight", {8}, true);
    for (int layer = 0; layer < 4; ++layer) {
        const auto p = "blk." + std::to_string(layer) + ".";
        add(p + "attn_norm.weight", {8}, true); add(p + "post_attention_norm.weight", {8}, true);
        if (layer != 3) {
            add(p + "attn_qkv.weight", {8,8}); add(p + "attn_gate.weight", {4,8});
            add(p + "ssm_alpha.weight", {2,8}); add(p + "ssm_beta.weight", {2,8});
            add(p + "ssm_out.weight", {8,4}); add(p + "ssm_conv1d.weight", {8,4});
            add(p + "ssm_a", {2}, false, true); add(p + "ssm_dt.bias", {2}); add(p + "ssm_norm.weight", {2}, true);
        } else {
            add(p + "attn_q.weight", {16,8}); add(p + "attn_k.weight", {4,8}); add(p + "attn_v.weight", {4,8});
            add(p + "attn_output.weight", {8,8}); add(p + "attn_q_norm.weight", {4}, true); add(p + "attn_k_norm.weight", {4}, true);
        }
        add(p + "ffn_gate.weight", {12,8}); add(p + "ffn_up.weight", {12,8}); add(p + "ffn_down.weight", {8,12});
    }
    const Json::Object metadata{{"general.architecture","qwen35"},{"qwen35.embedding_length",8},{"qwen35.feed_forward_length",12},
        {"qwen35.block_count",4},{"qwen35.attention.head_count",2},{"qwen35.attention.head_count_kv",1},
        {"qwen35.context_length",32},{"qwen35.rope.dimension_count",2},{"qwen35.rope.freq_base",10000},
        {"qwen35.attention.layer_norm_rms_epsilon",1e-6},{"qwen35.ssm.conv_kernel",4},{"qwen35.ssm.group_count",1},
        {"qwen35.ssm.time_step_rank",2},{"qwen35.ssm.state_size",2},{"qwen35.ssm.inner_size",4}};
    std::ofstream out(path, std::ios::binary); out.write("GGUF",4); integer(out,3,4); integer(out,tensors.size(),8); integer(out,metadata.size(),8);
    for (const auto& pair : metadata) {
        string(out,pair.first);
        if (pair.second.is_string()) { integer(out,8,4); string(out,pair.second.string()); }
        else { integer(out,12,4); const double value = pair.second.number(); uint64_t bits; std::memcpy(&bits,&value,8); integer(out,bits,8); }
    }
    uint64_t offset = 0;
    for (const auto& item : tensors) {
        string(out,item.first); const auto& shape = item.second.shape(); integer(out,shape.size(),4);
        for (auto it = shape.rbegin(); it != shape.rend(); ++it) integer(out,*it,8);
        integer(out,0,4); integer(out,offset,8); offset += (item.second.nbytes() + 31) / 32 * 32;
    }
    while (uint64_t(out.tellp()) % 32) out.put(0);
    for (const auto& item : tensors) {
        const auto bytes = item.second.raw_data(); out.write(reinterpret_cast<const char*>(bytes.data()),bytes.size());
        while (uint64_t(out.tellp()) % 32) out.put(0);
    }
    check(bool(out),"Cannot write tiny hybrid model fixture");
}
void near(const Tensor& a, const Tensor& b, float tolerance = 5e-5f) {
    check(a.shape() == b.shape(),"Hybrid result shape mismatch");
    const auto x = cast(a,DType::Float32).data(), y = cast(b,DType::Float32).data();
    for (size_t i = 0; i < x.size(); ++i) check(std::isfinite(x[i]) && std::abs(x[i]-y[i]) < tolerance,"Hybrid result mismatch");
}
}
int main(int argc, char** argv) {
    try {
        check(argc == 3,"Usage: elman_qwen35_tests BACKEND DIRECTORY");
        const auto backend = parse_backend(argv[1]); if (!backend_available(backend)) return 77;
        const auto directory = std::filesystem::u8path(argv[2]); std::filesystem::create_directories(directory);
        const auto path = directory / "tiny.gguf"; fixture(path);
        const Qwen35 model(path,backend,DType::Float32), cpu(path,Backend::CPU,DType::Float32);
        const auto ids = Tensor::from_int_data({1,5},{1,2,3,4,5});
        const auto full = model.forward(ids);
        near(full.logits,cpu.forward(ids).logits);
        const auto first = model.forward(slice(ids,1,0,2));
        const auto last = model.forward(slice(ids,1,2,5),{},{},&first.cache);
        near(concat({first.logits,last.logits},1),full.logits);
        near(first.cache.layers[0].key,model.forward(slice(ids,1,0,2)).cache.layers[0].key);
        near(last.cache.layers[0].value,full.cache.layers[0].value);
        check(full.cache.length == 5 && full.cache.layers[0].key.shape() == Shape{1,3,8} &&
            full.cache.layers[0].value.shape() == Shape{1,2,2,2} && full.cache.layers[3].key.shape() == Shape{1,1,5,4},"Hybrid cache layout differs");
        near(model.forward(ids,{},{},nullptr,false,DecoderLogits::Last).logits,slice(full.logits,1,4,5));
        check(!model.forward(ids,{},{},nullptr,false,DecoderLogits::None).logits.defined(),"Logits selection was ignored");
        const Qwen35 half(path,backend,DType::Float16); near(half.forward(ids).logits,full.logits,.006f);
        const auto loaded = load_decoder(path.u8string(),backend,DType::Float32);
        check(loaded->config().recurrent_layers == 3 && loaded->config().cache_elements_per_token() == 8,
            "Hybrid factory cache accounting differs");
        near(loaded->forward(ids).logits,full.logits);
        GenerationOptions options; options.max_context = 32; options.max_new_tokens = 3;
        const auto complete = generate(*loaded,{1,2,3,4,5},options);
        options.prefill_chunk_size = 2;
        check(generate(*loaded,{1,2,3,4,5},options).ids == complete.ids,"Hybrid generation differs across prompt chunks");
        const auto weights = plan_decoder_weight_load(path.u8string(),backend,DType::Float32);
        check(weights.parameter_bytes > 0,"Hybrid weight memory was omitted");
        const auto memory = plan_decoder_generation(loaded->config(),backend,DType::Float32,5,3,false);
        check(memory.kv_cache_bytes >= 4 * (3 * 2 * 2 * 2 + 3 * 3 * 8 + 8 * 8),"Hybrid cache memory was underestimated");
        bool rejected = false;
        try { model.forward(Tensor::from_int_data({1,1},{16})); } catch (const std::invalid_argument&) { rejected = true; }
        check(rejected,"Out-of-range hybrid token accepted");
        std::cout << "Qwen3.5 component checks passed\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
