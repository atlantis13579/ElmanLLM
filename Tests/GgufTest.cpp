#include "IO/Gguf.h"
#include "Text/HuggingFaceTokenizer.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void integer(std::ostream& out, uint64_t value, int bytes) { for (int i = 0; i < bytes; ++i) out.put(char(value >> (i * 8))); }
void string(std::ostream& out, const std::string& value) { integer(out,value.size(),8); out.write(value.data(),value.size()); }
void number(std::ostream& out, double value) { uint64_t bits; std::memcpy(&bits,&value,8); integer(out,bits,8); }
void fixture(const std::filesystem::path& path) {
    const char* chat =
#include "Text/Qwen35ChatTemplate.inc"
    ;
    Json::Array vocabulary, types;
    for (char16_t code : bytelevel_codepoints()) { vocabulary.emplace_back(utf8(std::u16string(1,code))); types.emplace_back(1); }
    vocabulary.emplace_back("Hi"); types.emplace_back(1);
    for (const char* token : {"<|endoftext|>","<|im_start|>","<|im_end|>","<think>","</think>"}) {
        vocabulary.emplace_back(token); types.emplace_back(3);
    }
    const Json::Object metadata{{"general.architecture","qwen35"},{"tokenizer.ggml.model","gpt2"},{"tokenizer.ggml.pre","qwen35"},
        {"tokenizer.ggml.tokens",vocabulary},{"tokenizer.ggml.token_type",types},{"tokenizer.ggml.merges",Json::Array{"H i"}},
        {"tokenizer.ggml.eos_token_id",257},{"tokenizer.ggml.padding_token_id",257},{"tokenizer.chat_template",chat},
        {"test.numeric_array",Json::Array{1.5,-2.75}},{"test.negative",-3}};
    std::ofstream out(path,std::ios::binary); out.write("GGUF",4); integer(out,3,4); integer(out,1,8); integer(out,metadata.size(),8);
    for (const auto& pair : metadata) {
        string(out,pair.first); const auto& value = pair.second;
        if (value.is_array()) {
            integer(out,9,4); const auto& items = value.array();
            const bool strings = items.front().is_string(); integer(out,strings ? 8 : 12,4); integer(out,items.size(),8);
            for (const auto& item : items) { if (strings) string(out,item.string()); else number(out,item.number()); }
        } else if (value.is_string()) { integer(out,8,4); string(out,value.string()); }
        else { integer(out,12,4); number(out,value.number()); }
    }
    string(out,"weight"); integer(out,2,4); integer(out,64,8); integer(out,2,8); integer(out,8,4); integer(out,0,8);
    while (uint64_t(out.tellp()) % 32) out.put(0);
    for (int block = 0; block < 4; ++block) {
        integer(out,block == 0 ? 0x3800 : block == 1 ? 0xb800 : block == 2 ? 0 : 0x3c00,2);
        for (int i = 0; i < 32; ++i) out.put(char(i * 8 - 128));
    }
    check(bool(out),"Cannot write GGUF fixture");
}
void near(const Tensor& a,const Tensor& b) {
    const auto x = cast(a,DType::Float32).data(), y = cast(b,DType::Float32).data();
    check(x.size() == y.size(),"GGUF shape mismatch");
    for (size_t i = 0; i < x.size(); ++i) check(std::abs(x[i]-y[i]) < 1e-6,"GGUF affine import changed values");
}
}
int main(int argc,char** argv) {
    try {
        check(argc == 3,"Usage: elman_gguf_tests BACKEND DIRECTORY");
        const auto backend = parse_backend(argv[1]); if (!backend_available(backend)) return 77;
        const auto directory = std::filesystem::u8path(argv[2]); std::filesystem::create_directories(directory);
        const auto path = directory / "component.gguf"; fixture(path); const GgufFile file(path);
        check(file.number_arrays().at("test.numeric_array") == std::vector<double>{1.5,-2.75},"GGUF numeric array was lost");
        check(file.numbers().at("test.negative") == -3,"GGUF signed metadata was lost");
        const auto quantized = file.load_quantized("weight",backend);
        for (const auto dtype : {DType::Float32,DType::Float16,DType::BFloat16}) {
            near(quantized.dequantize(dtype),cast(file.load("weight",backend),dtype));
            auto workspace = Tensor::zeros(quantized.shape(),backend,false,dtype);
            quantized.dequantize_into(workspace);
            near(workspace,cast(file.load("weight",backend),dtype));
        }
        const HuggingFaceTokenizer tokenizer(path.u8string());
        check(tokenizer.encode("Hi",false) == std::vector<int32_t>{256},"GGUF BPE merge failed");
        const std::string original = "Hi e\xcc\x81 \xe4\xb8\xad\n";
        check(tokenizer.decode(tokenizer.encode(original,false)) == original,"GGUF Unicode input was normalized or lost");
        check(tokenizer.stop_ids() == std::vector<int32_t>{257,259},"ChatML stopping IDs were lost");
        const auto rendered = tokenizer.render_chat({Json::Object{{"role","user"},{"content","  Hi  "}}});
        check(rendered == "<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n","Incorrect IndexTranslate chat prompt");
        check(tokenizer.decode(tokenizer.encode(rendered,false)) == rendered,"ChatML special token roundtrip failed");
        std::filesystem::resize_file(path,std::filesystem::file_size(path)-1);
        bool rejected = false; try { GgufFile bad(path); } catch (const std::exception&) { rejected = true; }
        check(rejected,"Truncated GGUF was accepted");
        std::cout << "GGUF and ChatML component checks passed\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
