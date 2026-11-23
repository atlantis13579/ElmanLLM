#include "Text/TokenStream.h"

#include <iostream>
#include <stdexcept>

namespace {
size_t checks = 0;
void check(bool value) {
    if (!value) throw std::runtime_error("Text stream check failed at " + std::to_string(checks));
    ++checks;
}
void sequence(const std::vector<std::string>& stops, const std::vector<std::string>& chunks,
              const std::vector<std::string>& expected, const std::string& tail, const std::string& matched) {
    elman::TextStopFilter filter(stops);
    check(chunks.size() == expected.size());
    for (size_t i = 0; i < chunks.size(); ++i) {
        std::string value = filter.push(chunks[i]);
        elman::utf16(value);
        check(value == expected[i]);
    }
    check(filter.finish() == tail);
    check(filter.finish().empty());
    check(filter.stopped() == !matched.empty());
    check(filter.matched_stop() == matched);
    bool rejected = false;
    try { filter.push("after finish"); } catch (const std::logic_error&) { rejected = true; }
    check(rejected);
}
}

int main() {
    try {
        sequence({}, {"hello", "", " world"}, {"hello", "", " world"}, "", "");
        sequence({"STOP"}, {"one S", "T", "O", "P trailing", "ignored"}, {"one ", "", "", "", ""}, "", "STOP");
        sequence({"STOP"}, {"one ST", "uff"}, {"one ", "STuff"}, "", "");
        sequence({"STOP"}, {"one ST"}, {"one "}, "ST", "");
        sequence({"STOP"}, {"STOP"}, {""}, "", "STOP");
        sequence({"END", "STOP"}, {"xSTOP END"}, {"x"}, "", "STOP");
        sequence({"abc", "ab"}, {"abc"}, {""}, "", "abc");
        sequence({"abc", "b"}, {"a", "b", "c"}, {"", "a", ""}, "", "b");
        sequence({"aaaa"}, {"aaa", "aa"}, {"", ""}, "", "aaaa");
        sequence({"abc", "abd"}, {"ab", "d"}, {"", ""}, "", "abd");
        sequence({"abc", "abd"}, {"ab", "x"}, {"", "abx"}, "", "");
        sequence({u8"结束🚀"}, {u8"你好结", u8"束", u8"🚀后"}, {u8"你好", "", ""}, "", u8"结束🚀");
        sequence({u8"中文"}, {u8"中", u8"华"}, {"", u8"中华"}, "", "");
        sequence({std::string("\0x", 2)}, {std::string("a\0", 2), "xy"}, {"a", ""}, "", std::string("\0x", 2));
        // All chunkings at Unicode scalar boundaries must preserve a unique stop.
        const std::vector<std::string> scalars = {"x", u8"🚀", u8"结", u8"束", "y", "z"};
        for (unsigned mask = 0; mask < (1u << (scalars.size() - 1)); ++mask) {
            elman::TextStopFilter filter({u8"结束"});
            std::string chunk, output;
            for (size_t i = 0; i < scalars.size(); ++i) {
                chunk += scalars[i];
                if (i + 1 == scalars.size() || (mask & (1u << i))) {
                    std::string value = filter.push(chunk);
                    elman::utf16(value);
                    output += value;
                    chunk.clear();
                }
            }
            output += filter.finish();
            check(output == u8"x🚀" && filter.matched_stop() == u8"结束");
        }
        for (const std::string& stop : {std::string(), std::string("\xff", 1)}) {
            bool rejected = false;
            try { elman::TextStopFilter filter({stop}); } catch (const std::exception&) { rejected = true; }
            check(rejected);
        }
        elman::TextStopFilter filter;
        bool rejected = false;
        try { filter.push(std::string("\xe0\xa0", 2)); } catch (const std::exception&) { rejected = true; }
        check(rejected);
        check(filter.push("valid") == "valid");
        std::cout << "Text stream: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
