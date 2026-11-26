#include "Commands/Commands.h"
#include "Runtime/RunContext.h"
#include "IO/Checkpoint.h"
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Inference/LLM/Sampling.h"
#include "Text/CharacterTokenizer.h"

#include <iostream>
#include <stdexcept>

namespace elman {

int sample_command(int argc, char** argv) {
  if (argc < 3) throw std::invalid_argument("sample requires a training configuration");
  RunOptions options;
  std::string prompt;
  bool prompt_set = false;
  int count = -1, seed = -1;
  double temperature = -1;
  bool random_weights = false;
  for (int i = 3; i < argc; ++i) {
    const std::string key = argv[i];
    if (key == "--random-weights") { random_weights = true; continue; }
    if (++i == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[i];
    if (key == "--backend") options.backend = value;
    else if (key == "--checkpoint") options.checkpoint = value;
    else if (key == "--output") options.output = value;
    else if (key == "--prompt") { prompt = value; prompt_set = true; }
    else if (key == "--max-new-tokens") count = std::stoi(value);
    else if (key == "--seed") seed = std::stoi(value);
    else if (key == "--temperature") temperature = std::stod(value);
    else throw std::invalid_argument("Unknown sample option: " + key);
  }
  Configuration config(argv[2], options);
  CharacterTokenizer tokenizer(config.definition.at("tokenizer"));
  if (!config.model->language() || tokenizer.size() != size_t(config.model->output_classes()))
    throw std::invalid_argument("sample requires a character language model");
  const std::string checkpoint = options.checkpoint.empty()
      ? config.output("best_checkpoint", config.output("checkpoint", "../Intermediate/checkpoint.elman"))
      : options.checkpoint;
  if (random_weights && !options.checkpoint.empty())
    throw std::invalid_argument("--random-weights cannot be combined with --checkpoint");
  if (!random_weights) load_checkpoint(checkpoint, *config.model, config.definition);
  const Json settings = config.json.contains("sampling") ? config.json.at("sampling") : Json(Json::Object{});
  if (count == -1) count = settings.int_or("max_new_tokens", 1000);
  if (seed == -1) seed = settings.int_or("seed", 1337);
  if (count <= 0 || seed < 0) throw std::invalid_argument("Invalid sample length/seed");
  SamplingOptions sampling;
  sampling.temperature = temperature == -1 ? settings.number_or("temperature", 0.8) : temperature;
  sampling.top_k = settings.int_or("top_k", 0);
  validate_sampling(sampling);
  Json::Array prompts;
  if (prompt_set) prompts.emplace_back(prompt);
  else if (settings.contains("prompts")) prompts = settings.at("prompts").array();
  else prompts.emplace_back("\n");
  AllocationPool pool(config.backend, 1024ull * 1024 * 1024);
  ExecutionStream stream(config.backend);
  StreamScope stream_scope(stream);
  Json::Array samples;
  for (size_t sample = 0; sample < prompts.size(); ++sample) {
    const std::string prefix = prompts[sample].string();
    auto tokens = tokenizer.encode(prefix);
    if (tokens.empty()) throw std::invalid_argument("Sample prompt must not be empty");
    SamplingRng rng(uint64_t(seed) + sample);
    std::string generated;
    for (int step = 0; step < count; ++step) {
      const size_t length = std::min(tokens.size(), size_t(config.model->sample_shape().at(0)));
      const std::vector<int32_t> window(tokens.end() - length, tokens.end());
      const Tensor input = Tensor::from_int_data({1, int(length)}, window, config.backend);
      Tensor logits = config.model->forward(input, false, config.seed);
      // Download only the final position; the model uses a sliding context.
      logits = reshape(slice(logits, 1, int(length) - 1, int(length)), {1, int(tokenizer.size())});
      const int32_t token = sample_tokens(logits, input, sampling, rng).at(0);
      tokens.push_back(token);
      generated += tokenizer.decode(token);
    }
    std::cout << prefix << generated << "\n\n";
    samples.emplace_back(Json::Object{{"prompt", prefix}, {"generated", generated},
                                     {"seed", double(uint64_t(seed) + sample)}});
  }
  const std::string output = options.output.empty() ? config.output("samples", "../Intermediate/samples.json") : options.output;
  Json(Json::Object{{"format", "elman.character_samples"}, {"model", config.model->name()},
                   {"checkpoint", random_weights ? "" : checkpoint}, {"random_weights", random_weights},
                   {"backend", backend_name(config.backend)},
                   {"temperature", sampling.temperature}, {"top_k", sampling.top_k},
                   {"max_new_tokens", count}, {"samples", std::move(samples)}}).write(output);
  return 0;
}

}  // namespace elman
