#include "ChatService.h"
#include <filesystem>
#include <iostream>
#include <thread>

using namespace elman;
namespace {
size_t checks = 0;
void check(bool value) { if (!value) throw std::runtime_error("Chat output check failed: " + std::to_string(checks)); ++checks; }
template<class F> void rejects(F action, const char* code) {
    bool caught = false; try { action(); } catch (const ChatError& error) { caught = error.code == code; } check(caught);
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_chat_tests TOKENIZER");
        HuggingFaceTokenizer tokenizer(argv[1]);
        auto definition = Json::read((std::filesystem::path(argv[1]) / "tokenizer.json").string());
        auto byte_id = [&](int byte) {
            auto visible = [](int b) { return (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || b >= 174; };
            int code = byte;
            if (!visible(byte)) { code = 256; for (int b = 0; b < byte; ++b) if (!visible(b)) ++code; }
            return definition.at("model").at("vocab").at(utf8(std::u16string(1, char16_t(code)))).integer();
        };
        auto baseline = memory_stats(Backend::CPU).workspace_bytes;
        ChatServiceOptions settings; settings.output_bytes = 64; settings.stream_bytes = 16; settings.stream_chunks = 4;
        ChatInput input; input.stream = true;
        GenerationResult complete; complete.finish_reason = "length";
        {
            ChatOutput output(tokenizer, input, settings);
            check(output.accept(byte_id(0xf0)) && output.accept(byte_id(0x9f)));
            check(output.pop(std::chrono::milliseconds(0)).empty());
            std::string tail; auto result = output.finish(complete, tail);
            check(tail == u8"�" && result.text == tail && result.finish_reason == "length");
        }
        {
            auto stopped = input; stopped.stops = {u8"�"};
            ChatOutput output(tokenizer, stopped, settings);
            check(output.accept(byte_id(0xf0)) && output.accept(byte_id(0x9f)));
            std::string tail; auto result = output.finish(complete, tail);
            check(tail.empty() && result.text.empty() && result.finish_reason == "stop" && result.matched_stop == u8"�");
        }
        {
            auto stopped = input; stopped.stops = {u8"🚀"};
            ChatOutput output(tokenizer, stopped, settings);
            check(output.accept(byte_id('a')));
            for (int byte : {0xf0, 0x9f, 0x9a}) check(output.accept(byte_id(byte)));
            check(!output.accept(byte_id(0x80)));
            check(output.pop(std::chrono::milliseconds(0)) == "a");
            check(output.pop(std::chrono::milliseconds(0)).empty());
            std::string tail; auto result = output.finish(complete, tail);
            check(tail.empty() && result.text == "a" && result.finish_reason == "stop" && result.matched_stop == u8"🚀");
        }
        {
            auto stopped = input; stopped.stops = {"STOP"};
            ChatOutput output(tokenizer, stopped, settings);
            for (int byte : {'a', 'S', 'T'}) check(output.accept(byte_id(byte)));
            check(output.pop(std::chrono::milliseconds(0)) == "a");
            std::string tail; auto result = output.finish(complete, tail);
            check(tail == "ST" && result.text == "aST" && result.finish_reason == "length");
        }
        {
            auto bounded = settings; bounded.stream_chunks = 1;
            ChatOutput slow(tokenizer, input, bounded), other(tokenizer, input, bounded);
            check(slow.accept(byte_id('a')));
            rejects([&] { slow.accept(byte_id('b')); }, "output_backpressure");
            check(other.accept(byte_id('c')) && other.pop(std::chrono::milliseconds(0)) == "c");
            slow.abandon(); check(slow.pop(std::chrono::milliseconds(0)).empty());
            rejects([&] { slow.accept(byte_id('d')); }, "client_closed");
        }
        {
            auto bounded = settings; bounded.stream_bytes = 1;
            ChatOutput output(tokenizer, input, bounded);
            check(output.accept(byte_id('a')));
            rejects([&] { output.accept(byte_id('b')); }, "output_backpressure");
        }
        {
            auto bounded = settings; bounded.output_bytes = 1;
            auto plain = input; plain.stream = false;
            ChatOutput output(tokenizer, plain, bounded);
            check(output.accept(byte_id('a')));
            rejects([&] { output.accept(byte_id('b')); }, "output_limit");
        }
        {
            ChatOutput output(tokenizer, input, settings); std::string received;
            std::thread consumer([&] { received = output.pop(std::chrono::seconds(2)); });
            output.accept(byte_id('x')); consumer.join(); check(received == "x");
        }
        check(memory_stats(Backend::CPU).workspace_bytes == baseline);
        std::cout << "Chat output: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
