#pragma once

#include "Runtime/RunOptions.h"
#include <string>

namespace elman {
int generate_command(int argc, char** argv);
int sample_command(int argc, char** argv);
int infer_tensors_command(int argc, char** argv);
int transcribe_command(int argc, char** argv);
int tokenizer_command(const std::string& command, int argc, char** argv);
int train(const std::string& configuration, const RunOptions& options = {});
}
