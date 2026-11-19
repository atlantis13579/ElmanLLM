#pragma once

#include <iostream>
#include <mutex>
#include <string_view>

namespace elman {

inline void server_status(std::string_view scope, std::string_view message) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::cout << '[' << scope << "] " << message << '\n' << std::flush;
}

} // namespace elman
