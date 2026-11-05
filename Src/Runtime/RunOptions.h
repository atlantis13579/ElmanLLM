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

}  // namespace elman
