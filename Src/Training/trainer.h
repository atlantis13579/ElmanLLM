#pragma once

#include <string>

namespace elman {

struct RunOptions {
    std::string backend;
    std::string checkpoint;
    std::string resume;
    std::string output;
    int steps = 0;
};

int train(const std::string& configuration, const RunOptions& options = {});
int infer(const std::string& configuration, const RunOptions& options = {});
int validate_json(const std::string& path);

}
