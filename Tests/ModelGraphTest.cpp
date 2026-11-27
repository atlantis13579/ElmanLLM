#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

#include "IO/Checkpoint.h"
#include "Inference/InferenceSession.h"
#include "Models/GraphModel.h"

using namespace elman;
namespace {
void check(bool value, const char* message) {
  if (!value) {
    throw std::runtime_error(message);
  }
}
template <class F>
void rejects(F action) {
  try {
    action();
  } catch (const std::exception&) {
    return;
  }
  throw std::runtime_error("Invalid graph accepted");
}
Json graph_definition() {
  return Json::parse(
      R"({"format":"elman.model","version":2,"name":"Branch","task":"regression","output_features":2,
    "inputs":[{"name":"x","dtype":"float32","shape":[-1,2]},{"name":"y","dtype":"float32","shape":[-1,2]}],
    "outputs":[{"name":"prediction","value":"sum","dtype":"float32","shape":[-1,2]},
               {"name":"indices","value":"indices","dtype":"int32","shape":[-1,2]}],
    "layers":[{"name":"sum","class_name":"add","inputs":["dense","y"]},
              {"name":"dense","class_name":"Linear","config":{"in_features":2,"out_features":2},"inputs":["x"]},
              {"name":"sort","class_name":"sort","inputs":["sum"],"outputs":["sorted","indices"]}]})");
}
void graph_tests(Backend backend, const std::filesystem::path& directory) {
  auto definition = graph_definition();
  auto model = create_model(definition, backend, 10);
  auto* graph = dynamic_cast<GraphModel*>(model.get());
  graph->parameters()[0].value.assign({1, 2, 3, 4});
  graph->parameters()[1].value.assign({0, 0});
  nn::TensorMap input{{"x", Tensor::from_data({1, 2}, {2, 1}, backend)},
                      {"y", Tensor::from_data({1, 2}, {1, 2}, backend)}};
  auto result = graph->forward(input, true);
  check(result.at("prediction").data() == std::vector<float>({5, 12}),
        "Topological branch execution");
  sum_all(result.at("prediction")).backward();
  check(graph->parameters()[0].value.grad() == std::vector<float>({2, 1, 2, 1}),
        "Graph gradient");
  check(result.at("indices").int_data() == std::vector<int32_t>({0, 1}),
        "Multiple kernal outputs");
  const auto checkpoint = directory / "branch.elman";
  const auto path = directory / "branch.json";
  definition.write(path.string());
  TrainingState state;
  state.optimizer = "adam";
  for (const auto& parameter : graph->parameters()) {
    state.first_moment.emplace_back(parameter.value.numel(), 0.0f);
    state.second_moment.emplace_back(parameter.value.numel(), 0.0f);
  }
  save_checkpoint(checkpoint.string(), *graph, definition, state);
  auto session = load_model_session(path.string(), checkpoint.string(), backend,
                                    backend == Backend::CPU ? -1 : 0);
  auto inference = session.run(input);
  check(inference.at("prediction").data() == result.at("prediction").data(),
        "Named session ports");
  check(!inference.at("prediction").requires_grad(),
        "Inference graph disabled");
  auto malformed = definition.object();
  auto layers = malformed.at("layers").array();
  auto bad_node = layers[1].object();
  bad_node["inputs"] = Json::Array{"sum"};
  layers[1] = bad_node;
  malformed["layers"] = layers;
  rejects([&] { create_model(malformed, backend); });
  malformed = definition.object();
  layers = malformed.at("layers").array();
  bad_node = layers[0].object();
  bad_node["inputs"] = Json::Array{"missing", "y"};
  layers[0] = bad_node;
  malformed["layers"] = layers;
  rejects([&] { create_model(malformed, backend); });
  bad_node["inputs"] = Json::Array{"y"};
  layers[0] = bad_node;
  malformed["layers"] = layers;
  rejects([&] { create_model(malformed, backend); });
  bad_node["class_name"] = "NotAnKernal";
  layers[0] = bad_node;
  malformed["layers"] = layers;
  rejects([&] { create_model(malformed, backend); });
  rejects([&] {
    nn::validate_operation(
        "Linear",
        Json::Object{{"in_features", 2}, {"out_features", 2}, {"typo", 1}}, 1,
        1);
  });
  rejects([&] { graph->forward(nn::TensorMap{{"x", input.at("x")}}, false); });
  check(nn::kernal_catalog().at("entries").object().size() > 100,
        "Kernal catalog coverage");
}
void state_tests(Backend backend, const std::filesystem::path& directory) {
  const auto definition = Json::parse(
      R"({"format":"elman.model","version":2,"name":"State","task":"regression","output_features":2,
    "inputs":[{"name":"input","dtype":"float32","shape":[-1,2,2,2]}],
    "outputs":[{"name":"output","value":"norm","dtype":"float32","shape":[-1,2,2,2]}],
    "layers":[{"name":"norm","class_name":"BatchNorm2d","inputs":["input"],"config":{"channels":2}}]})");
  auto model = create_model(definition, backend);
  const auto input = Tensor::normal({2, 2, 2, 2}, 3, 1, 5, backend);
  const auto before = model->checkpoint_buffers()[0].value.data();
  sum_all(model->forward(input, true)).backward();
  check(before != model->checkpoint_buffers()[0].value.data(),
        "Training updates running statistics");
  const auto evaluation = model->forward(input, false).data();
  const auto checkpoint = directory / "state.elman";
  TrainingState state;
  state.optimizer = "adam";
  for (const auto& parameter : model->parameters()) {
    state.first_moment.emplace_back(parameter.value.numel(), 0.0f);
    state.second_moment.emplace_back(parameter.value.numel(), 0.0f);
  }
  save_checkpoint(checkpoint.string(), *model, definition, state);
  auto restored = create_model(definition, backend, 900);
  load_checkpoint(checkpoint.string(), *restored, definition);
  check(evaluation == restored->forward(input, false).data(),
        "Running statistics restored");
}

void subgraph_tests(Backend backend, const std::filesystem::path& directory) {
  const auto definition = Json::parse(R"({
    "format":"elman.model","version":2,"name":"Scoped","task":"regression","output_features":2,
    "parameters":{"factor":999},
    "inputs":[{"name":"x","dtype":"float32","shape":[-1,2]}],
    "outputs":[{"name":"out","value":"result","dtype":"float32","shape":[-1,2]}],
    "definitions":{
      "Scale":{"class_name":"Graph","parameters":{"factor":null},"config":{
        "inputs":["x"],"outputs":["scaled"],"layers":[
          {"name":"scaled","class_name":"mul_scalar","inputs":["x"],"config":{"b":"$factor"}}]}},
      "Double":{"class_name":"Sequence","parameters":{"factor":2},"config":{"layers":[
        {"name":"first","class_name":"Subgraph","config":{"ref":"Scale","parameters":{"factor":"$factor"}}},
        {"name":"second","class_name":"Subgraph","config":{"ref":"Scale","parameters":{"factor":"$factor"}}}]}}
    },
    "layers":[{"name":"result","class_name":"Sequence","inputs":["x"],"config":{"repeat":2,"layers":[
      {"name":"block","class_name":"Subgraph","config":{"ref":"Double","parameters":{"factor":3}}}]}}]
  })");
  auto model = create_model(definition, backend);
  const auto input = Tensor::from_data({1, 2}, {1, 2}, backend, true);
  const auto output = model->forward(input, true);
  check(output.data() == std::vector<float>({81, 162}),
        "Nested subgraph local parameter scope and repeats");
  sum_all(output).backward();
  check(input.grad() == std::vector<float>({81, 81}),
        "Nested subgraph gradient");

  auto malformed = definition.object();
  auto definitions = malformed.at("definitions").object();
  auto scale = definitions.at("Scale").object();
  scale["config"] =
      Json::parse(R"({"inputs":["x"],"outputs":["loop"],"layers":[
    {"name":"loop","class_name":"Subgraph","inputs":["x"],"config":{"ref":"Double"}}]})");
  definitions["Scale"] = scale;
  malformed["definitions"] = definitions;
  rejects([&] { create_model(malformed, backend); });
  for (const auto& config :
       {Json::Object{{"ref", "Missing"}}, Json::Object{{"ref", "Scale"}},
        Json::Object{{"ref", "Scale"},
                     {"parameters", Json::Object{{"typo", 3}}}}}) {
    malformed = definition.object();
    malformed["layers"] = Json::Array{Json::Object{{"name", "result"},
                                                   {"class_name", "Subgraph"},
                                                   {"inputs", Json::Array{"x"}},
                                                   {"config", config}}};
    rejects([&] { create_model(malformed, backend); });
  }

  const auto includes = directory / "includes";
  std::filesystem::create_directories(includes);
  definition.at("definitions").write((includes / "library.json").string());
  Json(Json::Object{{"$include", "library.json"}})
      .write((includes / "nested.json").string());
  auto file_definition = definition.object();
  file_definition["definitions"] =
      Json::Object{{"$include", "includes/nested.json"}};
  const auto file = directory / "scoped.json";
  Json(file_definition).write(file.string());
  const auto embedded = read_model_definition(file);
  check(embedded.dump(0) == definition.dump(0),
        "Includes resolve relative to each containing file");
  embedded.write((directory / "portable.json").string());
  Json(Json::Object{{"$include", "nested.json"}})
      .write((includes / "library.json").string());
  rejects([&] { read_model_definition(file); });
  check(
      create_model(read_model_definition(directory / "portable.json"), backend)
              ->forward(input, false)
              .data() == output.data(),
      "Expanded JSON remains usable after external includes change");
  Json(Json::Object{{"$include", "missing.json"}})
      .write((includes / "nested.json").string());
  rejects([&] { read_model_definition(file); });
  Json(Json::Object{{"$include", "library.json"}, {"ambiguous_override", 1}})
      .write((includes / "nested.json").string());
  rejects([&] { read_model_definition(file); });
}

void compact_catalog_tests() {
  const auto& catalog = nn::kernal_catalog();
  check(catalog.at("version").integer() == 2, "Compact catalog version");
  nn::validate_operation(
      "Linear", Json::Object{{"in_features", 2}, {"out_features", 3}}, 1, 1);
  rejects([] {
    nn::validate_operation("Linear", Json::Object{{"out_features", 3}}, 1, 1);
  });
  nn::validate_operation("linear", Json::Object{}, 2, 1);
  nn::validate_operation("linear", Json::Object{}, 3, 1);
  rejects([] { nn::validate_operation("linear", Json::Object{}, 1, 1); });
  rejects([] { nn::validate_operation("linear", Json::Object{}, 4, 1); });
  nn::validate_operation("concat", Json::Object{{"dim", 0}}, 5, 1);
  rejects(
      [] { nn::validate_operation("concat", Json::Object{{"dim", 0}}, 0, 1); });
  nn::validate_operation("split", Json::Object{{"size", 2}}, 1, 4);
  nn::validate_operation("sort", Json::Object{}, 1, 2);
  rejects([] { nn::validate_operation("sort", Json::Object{}, 1, 1); });
  nn::validate_operation("Parameter",
                         Json::Object{{"shape", Json::Array{2, 3}}}, 0, 1);
  rejects([] { nn::validate_operation("Parameter", Json::Object{}, 0, 1); });
}

}  // namespace
int main(int argc, char** argv) {
  try {
    const auto backend = parse_backend(argc > 1 ? argv[1] : "cpu");
    if (!backend_available(backend)) {
      return 77;
    }
    const auto directory = std::filesystem::path(argv[2]);
    std::filesystem::create_directories(directory);
    graph_tests(backend, directory);
    state_tests(backend, directory);
    compact_catalog_tests();
    subgraph_tests(backend, directory);
    std::cout << "Model graph tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
