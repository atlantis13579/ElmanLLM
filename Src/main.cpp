#include <iostream>
#include <stdexcept>

#include "Core/Backend.h"
#include "Inference/Inference.h"
#include "Training/Trainer.h"

namespace {
void usage() {
  std::cout
      << "Elman self-contained deep-learning framework\n"
      << "Usage:\n"
      << "  elman devices\n"
      << "  elman train <training.json> [--backend cpu|cuda] [--steps N]\n"
      << "        [--checkpoint FILE] [--resume FILE] [--output metrics.json]\n"
      << "  elman infer <training.json> [--backend DEVICE] [--checkpoint FILE] "
         "[--output predictions.json]\n"
      << "  elman validate-json <file.json>\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 1;
    }
    std::string command = argv[1];
    if (command == "--help" || command == "help" || command == "-h") {
      usage();
      return 0;
    }
    if (command == "devices") {
      if (argc != 2) {
        throw std::runtime_error("devices takes no arguments");
      }
      for (auto backend : {elman::Backend::CPU, elman::Backend::CUDA}) {
        std::cout << elman::backend_name(backend) << ": "
                  << (elman::backend_available(backend) ? "available"
                                                        : "unavailable")
                  << '\n';
      }
      return 0;
    }
    if (argc < 3) {
      throw std::runtime_error("Command requires a JSON file");
    }
    if (command == "validate-json") {
      if (argc != 3) {
        throw std::runtime_error("validate-json takes one file argument");
      }
      return elman::validate_json(argv[2]);
    }
    elman::RunOptions options;
    for (int i = 3; i < argc; ++i) {
      std::string key = argv[i];
      if (i + 1 == argc) {
        throw std::runtime_error("Missing value for " + key);
      }
      std::string value = argv[++i];
      if (key == "--backend") {
        options.backend = value;
      } else if (key == "--checkpoint") {
        options.checkpoint = value;
      } else if (key == "--resume") {
        options.resume = value;
      } else if (key == "--output") {
        options.output = value;
      } else if (key == "--steps") {
        size_t consumed = 0;
        options.steps = std::stoi(value, &consumed);
        if (consumed != value.size() || options.steps <= 0) {
          throw std::runtime_error("--steps requires a positive integer");
        }
      } else {
        throw std::runtime_error("Unknown argument: " + key);
      }
    }
    if (command == "train") {
      return elman::train(argv[2], options);
    }
    if (command == "infer") {
      if (!options.resume.empty() || options.steps) {
        throw std::runtime_error("infer does not accept --resume or --steps");
      }
      return elman::infer(argv[2], options);
    }
    throw std::runtime_error("Unknown command: " + command);
  } catch (const std::exception& error) {
    std::cerr << "Elman error: " << error.what() << '\n';
    return 1;
  }
}
