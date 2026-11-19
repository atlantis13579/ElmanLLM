#include <iostream>
#include <stdexcept>

#include "Core/Backend.h"
#include "Core/Device.h"
#include "Runtime/DiffusionCommand.h"
#include "Inference/ModelEvaluation.h"
#include "Inference/InferenceSession.h"
#include "Inference/LLM/Generation.h"
#include "Training/Trainer.h"
#include "Text/Tokenizer.h"

#include "Web/Server.h"

namespace {
void usage() {
  std::cout
      << "Elman self-contained deep-learning framework\n"
      << "Usage:\n"
      << "  elman devices\n"
      << "  elman train <training.json> [--backend cpu|cuda|metal] [--steps N]\n"
      << "        [--checkpoint FILE] [--resume FILE] [--output metrics.json]\n"
      << "  elman infer <training.json> [--backend DEVICE] [--checkpoint FILE] "
         "[--output predictions.json]\n"
      << "  elman infer-tensors <model-or-graph> --input INPUT --output OUTPUT.safetensors\n"
      << "        [--checkpoint FILE] [--signature NAME] [--backend cpu|cuda|metal]\n"
      << "  elman diffusion <model-directory> --prompt TEXT --output IMAGE.png\n"
      << "        [--backend cpu|cuda|metal] [--dtype TYPE] [--steps N] [--seed N]\n"
      << "  elman tokenizer-train <tokenizer-training.json>\n"
      << "  elman tokenizer-encode <tokenizer.json> --input text.txt --output ids.json\n"
      << "  elman tokenizer-decode <tokenizer.json> --input ids.json --output text.txt\n";
  std::cout << "  elman generate <model-directory> --input prompts.json [--backend cpu|cuda|metal]\n"
               "        [--dtype float32|float16|bfloat16] [--max-new-tokens N]\n"
               "        [--max-context N] [--stop TEXT] [--stream] [--output generations.json]\n"
               "        [--sample] [--seed N] [--temperature X] [--top-k N] [--top-p X]\n"
               "        [--repetition-penalty X] [--memory-limit-mib N] [--memory-reserve-mib N]\n"
               "        [--no-memory-reuse] [--synchronous] [--no-pinned-staging] [--no-fusion] [--paged-cache]\n"
               "        [--prefill-chunk-size N] [--full-logits]\n"
               "        [--no-normalization-fusion] [--no-attention-fusion] [--no-tiled-attention]\n"
               "        [--no-packed-projections] [--mla-expanded] [--devices 0,1,...] [--cuda-graph]\n";
  std::cout << "        [--simulate-devices N] [--tensor-parallel]\n"
               "  elman --server-config <server.json>\n";
}
}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc < 2) {
      usage();
      return 1;
    }
    std::string command = argv[1];
    if (command.rfind("tokenizer-", 0) == 0) {
      return elman::tokenizer_command(command, argc, argv);
    }
    if (command == "generate") return elman::generate_command(argc, argv);
    if (command == "infer-tensors") return elman::infer_tensors_command(argc, argv);
    if (command == "diffusion") return elman::diffusion_command(argc, argv);
    if (command == "--server-config") {
      if (argc != 3) throw std::invalid_argument("--server-config requires one JSON file");
      elman::run_server(std::filesystem::u8path(argv[2]));
      return 0;
    }
    if (command == "--help" || command == "help" || command == "-h") {
      usage();
      return 0;
    }
    if (command == "devices") {
      if (argc != 2) {
        throw std::runtime_error("devices takes no arguments");
      }
      for (auto backend : {elman::Backend::CPU, elman::Backend::CUDA, elman::Backend::Metal}) {
        std::cout << elman::backend_name(backend) << ": "
                  << (elman::backend_available(backend) ? "available"
                                                        : "unavailable")
                  << '\n';
        if (elman::backend_available(backend)) for (const auto& device : elman::devices(backend))
          std::cout << "  " << device.ordinal << ": " << device.name << " (" << device.total_bytes << " bytes)\n";
      }
      return 0;
    }
    if (command != "train" && command != "infer") {
      throw std::runtime_error("Unknown command: " + command);
    }
    if (argc < 3) {
      throw std::runtime_error("Command requires a JSON file");
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
    if (!options.resume.empty() || options.steps) {
      throw std::runtime_error("infer does not accept --resume or --steps");
    }
    return elman::infer(argv[2], options);
  } catch (const std::exception& error) {
    std::cerr << "Elman error: " << error.what() << '\n';
    return 1;
  }
}
