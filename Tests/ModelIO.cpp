#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <stdexcept>

#include "IO/Checkpoint.h"
#include "Inference/ModelEvaluation.h"
#include "Runtime/RunContext.h"

namespace {
int checks = 0;
void check(bool condition, const std::string& message) {
  ++checks;
  if (!condition) {
    throw std::runtime_error(message);
  }
}
void rejects(const std::function<void()>& operation,
             const std::string& message) {
  bool rejected = false;
  try {
    operation();
  } catch (const std::exception&) {
    rejected = true;
  }
  check(rejected, message);
}
void near(const std::vector<float>& actual, const std::vector<float>& expected,
          float tolerance, const std::string& message) {
  check(actual.size() == expected.size(), message + " size");
  for (size_t i = 0; i < actual.size(); ++i) {
    check(std::isfinite(actual[i]) &&
              std::abs(actual[i] - expected[i]) <= tolerance,
          message + " value " + std::to_string(i));
  }
}

struct TemporaryDirectory {
  std::filesystem::path path;
  TemporaryDirectory() {
    path = std::filesystem::temp_directory_path() /
           ("elman_model_io_" +
            std::to_string(std::chrono::high_resolution_clock::now()
                               .time_since_epoch()
                               .count()));
    if (!std::filesystem::create_directory(path)) {
      throw std::runtime_error("Cannot create unique model test directory");
    }
  }
  ~TemporaryDirectory() {
    std::error_code ignored;
    std::filesystem::remove_all(path, ignored);
  }
};

elman::Json token_definition() {
  return elman::Json::parse(R"({
    "format":"elman.model","version":2,"name":"TokenLookupFixture",
    "task":"language_modeling","output_features":8,
    "inputs":[{"name":"input","dtype":"int32","shape":[-1,3]}],
    "outputs":[{"name":"output","value":"lookup","dtype":"float32","shape":[-1,3,8]}],
    "layers":[{"name":"weight","class_name":"Parameter","config":{"shape":[8,8]},"inputs":[]},
      {"name":"lookup","class_name":"embedding","config":{},"inputs":["input","weight"]}]
  })");
}
elman::Json affine_definition(bool regression) {
  auto result = elman::Json::parse(R"({
    "format":"elman.model","version":2,"name":"AffineFixture","task":"classification",
    "parameters":{"input_features":2,"classes":2},"output_features":"$classes",
    "inputs":[{"name":"input","dtype":"float32","shape":[-1,"$input_features"]}],
    "outputs":[{"name":"output","value":"dense","dtype":"float32","shape":[-1,"$classes"]}],
    "layers":[{"name":"dense","class_name":"Linear",
      "config":{"in_features":"$input_features","out_features":"$classes"},"inputs":["input"]}]
  })").object();
  result["task"] = regression ? "regression" : "classification";
  return result;
}
elman::TrainingState state_for(const elman::Model& model) {
  elman::TrainingState result;
  result.optimizer = "adam";
  result.step = 7;
  for (const elman::Parameter& parameter : model.parameters()) {
    result.first_moment.emplace_back(parameter.value.numel(), 0.125f);
    result.second_moment.emplace_back(parameter.value.numel(), 0.25f);
  }
  return result;
}
std::vector<char> read_bytes(const std::filesystem::path& path) {
  std::ifstream file(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}
void write_bytes(const std::filesystem::path& path,
                 const std::vector<char>& bytes) {
  std::ofstream file(path, std::ios::binary);
  file.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!file) {
    throw std::runtime_error("Cannot write checkpoint test fixture");
  }
}

void json_tests() {
  using elman::Json;
  Json value = Json::parse(
      R"({"a":[1,-2.5e2,true,false,null,"line\nend"],"unicode":"\u4e2d\u6587\ud83d\ude00"})");
  check(value.at("a").at(0).integer() == 1, "JSON integer");
  check(value.at("a").at(1).number() == -250, "JSON exponent");
  check(value.at("a").at(2).boolean() && !value.at("a").at(3).boolean(),
        "JSON booleans");
  check(value.at("a").at(4).is_null(), "JSON null");
  check(value.at("a").at(5).string() == "line\nend", "JSON escape");
  check(value.at("unicode").string() ==
            "\xe4\xb8\xad\xe6\x96\x87\xf0\x9f\x98\x80",
        "JSON surrogate-pair UTF8");
  check(Json::parse(value.dump()).dump(0) == value.dump(0),
        "JSON serialization roundtrip");
  for (const char* const& invalid : {"",
                              "[1,]",
                              "{\"a\":1,}",
                              "{\"a\":1,\"a\":2}",
                              "01",
                              "-01",
                              ".5",
                              "1.",
                              "1e",
                              "1e+",
                              "NaN",
                              "Infinity",
                              "1e9999",
                              "true false",
                              "\"\\uD800\"",
                              "\"\\uDC00\"",
                              "\"\\uD800\\u1234\"",
                              "\"\\q\"",
                              "\"unterminated",
                              "\"\n\""}) {
    rejects([&] { Json::parse(invalid); },
            std::string("JSON must reject: ") + invalid);
  }
  rejects([] { Json::parse("2147483648").integer(); }, "JSON integer range");
  rejects([] { Json::parse("1.25").integer(); }, "JSON fractional integer");
  rejects([] { Json::parse("false").number(); }, "JSON strict types");
  rejects(
      [] { Json::parse(std::string(130, '[') + "0" + std::string(130, ']')); },
      "JSON nesting bound");
}

void affine_tests() {
  using namespace elman;
  for (bool regression : {true, false}) {
    TemporaryDirectory temporary;
    const std::filesystem::path model_path = temporary.path / "model.json";
    const std::filesystem::path data_path = temporary.path / "data.csv";
    const std::filesystem::path config_path = temporary.path / "run.json";
    const std::filesystem::path checkpoint_path = temporary.path / "model.elman";
    const std::filesystem::path prediction_path = temporary.path / "predictions.json";
    Json definition = affine_definition(regression);
    definition.write(model_path.string());
    std::unique_ptr<Model> model = create_model(definition, Backend::CPU, 7);
    check(model->regression() == regression, "Affine objective selection");
    check(model->parameters().size() == 2, "Affine parameter count");
    model->parameters()[0].value.assign({1, 3, 2, 4});
    model->parameters()[1].value.assign({0.5f, -0.5f});
    Tensor input = Tensor::from_data({2, 2}, {1, 2, -1, 1});
    Tensor prediction = model->forward(input, false);
    near(prediction.data(), {7.5f, 9.5f, 2.5f, 1.5f}, 0, "Affine known output");
    check(!prediction.requires_grad(),
          "Affine inference has no parameter graph");
    rejects([&] { model->forward(Tensor::zeros({2, 3}), false); },
            "Affine input width");
    rejects(
        [&] { model->forward(Tensor::from_int_data({1, 2}, {1, 2}), false); },
        "Affine input dtype");
    elman::Json::Object bad = definition.object();
    auto bad_parameters = bad.at("parameters").object();
    bad_parameters["input_features"] = 0;
    bad["parameters"] = bad_parameters;
    rejects([&] { create_model(Json(bad), Backend::CPU); },
            "Affine invalid feature count");
    bad = definition.object();
    bad_parameters = bad.at("parameters").object();
    bad_parameters["classes"] = regression ? 0 : 1;
    bad["parameters"] = bad_parameters;
    rejects([&] { create_model(Json(bad), Backend::CPU); },
            "Affine invalid output count");
    save_checkpoint(checkpoint_path.string(), *model, definition,
                    state_for(*model));
    {
      std::ofstream data(data_path);
      data << (regression ? "7.5,9.5,1,2\n1.5,-1.5,-1,1\n" : "1,1,2\n0,-1,1\n");
    }
    Json config(Json::Object{
        {"format", "elman.training"},
        {"version", 1},
        {"model_name", model->name()},
        {"model", "model.json"},
        {"backend", "cpu"},
        {"training", Json::Object{{"seed", 7}}},
        {"data",
         Json::Object{{"format", regression ? "regression_csv" : "tabular_csv"},
                      {"path", "data.csv"}}},
        {"output", Json::Object{{"checkpoint", "model.elman"}}}});
    config.write(config_path.string());
    Configuration context(config_path.string(), {});
    Dataset data(context);
    TensorBatch batch = data.batch({0, 1}, 0, 2, Backend::CPU);
    near(batch.first.data(), input.data(), 0, "Tabular feature order");
    check(batch.second.shape() == (regression ? Shape{2, 2} : Shape{2}),
          "Tabular target shape");
    check(batch.second.dtype() == (regression ? DType::Float32 : DType::Int32),
          "Tabular target dtype");
    Tensor loss = data.loss(model->forward(batch.first, true), batch.second);
    if (regression) {
      near(loss.data(), {2.5f}, 0, "Multi-output mean squared error");
    }
    loss.backward();
    for (const elman::Parameter& parameter : model->parameters()) {
      check(parameter.value.grad().size() == parameter.value.numel(),
            "Affine parameter gradient");
      for (float value : parameter.value.grad()) {
        check(std::isfinite(value), "Affine finite gradient");
      }
    }
    RunOptions options;
    options.output = prediction_path.string();
    check(infer(config_path.string(), options) == 0,
          "Affine checkpoint inference");
    Json report = Json::read(prediction_path.string());
    check(report.at("predictions").array().size() == 2,
          "Affine inference sample count");
    if (regression) {
      check(!report.contains("accuracy"), "Regression has no class accuracy");
      check(std::abs(report.at("rmse").number() - std::sqrt(2.5)) < 1e-6,
            "Regression inference RMSE");
      check(
          report.at("predictions").at(0).at("predicted").at(1).number() == 9.5,
          "Regression fractional prediction");
    } else {
      check(report.at("accuracy").number() == 1, "Logistic inference accuracy");
    }
    {
      std::ofstream invalid(data_path);
      invalid << (regression ? "1,2,3\n" : "2,1,2\n");
    }
    rejects([&] { Dataset invalid(context); }, "Invalid affine dataset");
  }
}

void checkpoint_tests() {
  using namespace elman;
  TemporaryDirectory temporary;
  const std::filesystem::path good_path = temporary.path / "valid.elman";
  const std::filesystem::path bad_path = temporary.path / "invalid.elman";
  elman::Json definition = token_definition();
  std::unique_ptr<Model> model = create_model(definition, Backend::CPU, 7);
  elman::TrainingState state = state_for(*model);
  save_checkpoint(good_path.string(), *model, definition, state);
  std::unique_ptr<Model> restored = create_model(definition, Backend::CPU, 70);
  TrainingState loaded;
  load_checkpoint(good_path.string(), *restored, definition, &loaded);
  check(loaded.step == 7 && loaded.optimizer == "adam",
        "Checkpoint optimizer metadata");
  for (size_t i = 0; i < model->parameters().size(); ++i) {
    near(restored->parameters()[i].value.data(),
         model->parameters()[i].value.data(), 0,
         "Checkpoint exact parameter restore");
    near(loaded.first_moment[i], state.first_moment[i], 0,
         "Checkpoint first moments");
    near(loaded.second_moment[i], state.second_moment[i], 0,
         "Checkpoint second moments");
  }
  Tensor input =
      Tensor::from_data({1, 3}, {1, 2, 3}, Backend::CPU, false, DType::Int32);
  near(restored->forward(input, false).data(),
       model->forward(input, false).data(), 0,
       "Checkpoint inference equivalence");
  save_checkpoint(good_path.string(), *model, definition, state);
  check(!std::filesystem::exists(good_path.string() + ".tmp"),
        "Checkpoint temporary file removed after replacement");
  std::vector<char> bytes = read_bytes(good_path);
  check(bytes.size() > 100, "Checkpoint nonempty");
  std::vector<char> corrupted = bytes;
  corrupted[0] ^= 1;
  write_bytes(bad_path, corrupted);
  rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); },
          "Checkpoint signature validation");
  corrupted = bytes;
  corrupted[8] = 2;
  write_bytes(bad_path, corrupted);
  rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); },
          "Checkpoint version validation");
  corrupted = bytes;
  corrupted[corrupted.size() - 1] ^= 1;
  write_bytes(bad_path, corrupted);
  std::vector<float> previous_values = restored->parameters()[0].value.data();
  for (float& value : previous_values) {
    value += 1;
  }
  restored->parameters()[0].value.assign(previous_values);
  rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); },
          "Checkpoint checksum validation");
  near(restored->parameters()[0].value.data(), previous_values, 0,
       "Invalid checkpoint does not mutate parameters");
  corrupted = bytes;
  corrupted.resize(corrupted.size() / 2);
  write_bytes(bad_path, corrupted);
  rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); },
          "Truncated checkpoint validation");
  corrupted = bytes;
  corrupted.push_back(0);
  write_bytes(bad_path, corrupted);
  rejects([&] { load_checkpoint(bad_path.string(), *restored, definition); },
          "Trailing checkpoint bytes validation");
  elman::Json::Object changed_definition = definition.object();
  changed_definition["fixture_revision"] = Json(2);
  rejects(
      [&] {
        load_checkpoint(good_path.string(), *restored,
                        Json(changed_definition));
      },
      "Checkpoint model configuration validation");
  changed_definition = definition.object();
  changed_definition["name"] = Json("OtherModel");
  std::unique_ptr<Model> other = create_model(Json(changed_definition), Backend::CPU, 7);
  rejects(
      [&] {
        load_checkpoint(good_path.string(), *other, Json(changed_definition));
      },
      "Checkpoint model name validation");
}

void inference_tests() {
  using namespace elman;
  TemporaryDirectory temporary;
  const std::filesystem::path definition_path = temporary.path / "model.json";
  const std::filesystem::path data_path = temporary.path / "samples.csv";
  const std::filesystem::path checkpoint_path = temporary.path / "weights.elman";
  const std::filesystem::path configuration_path = temporary.path / "run.json";
  const std::filesystem::path prediction_path = temporary.path / "predictions.json";

  elman::Json definition = token_definition();
  definition.write(definition_path.string());
  std::unique_ptr<Model> model = create_model(definition, Backend::CPU, 7);
  save_checkpoint(checkpoint_path.string(), *model, definition,
                  state_for(*model));
  {
    std::ofstream data(data_path);
    data << "1,2,3,4\n";
    check(static_cast<bool>(data), "Inference dataset fixture");
  }

  Json configuration(Json::Object{
      {"format", "elman.training"},
      {"version", 1},
      {"model_name", model->name()},
      {"model", definition_path.filename().string()},
      {"backend", "cpu"},
      {"data", Json::Object{{"format", "token_csv"},
                            {"path", data_path.filename().string()},
                            {"sequence_length", 3}}},
      {"training", Json::Object{{"seed", 7}}},
      {"output",
       Json::Object{{"checkpoint", checkpoint_path.filename().string()}}}});
  configuration.write(configuration_path.string());

  RunOptions options;
  options.output = prediction_path.string();
  check(infer(configuration_path.string(), options) == 0,
        "Independent inference entry point");
  Json result = Json::read(prediction_path.string());
  check(result.at("format").string() == "elman.inference",
        "Inference report format");
  check(result.at("model").string() == model->name(), "Inference model name");
  check(result.at("backend").string() == "cpu", "Inference backend");
  check(result.at("predictions").array().size() == 1, "Inference sample count");
  const Json& prediction = result.at("predictions").at(0);
  check(prediction.at("predicted").array().size() == 3,
        "Inference token count");
  check(prediction.at("target").at(0).integer() == 2 &&
            prediction.at("target").at(1).integer() == 3 &&
            prediction.at("target").at(2).integer() == 4,
        "Inference target tokens");
  check(std::isfinite(result.at("loss").number()), "Inference finite loss");

  Tensor input =
      Tensor::from_data({1, 3}, {1, 2, 3}, Backend::CPU, false, DType::Int32);
  std::vector<float> logits = model->forward(input, false).data();
  for (int token = 0; token < 3; ++token) {
    int expected = 0;
    for (int cls = 1; cls < model->output_classes(); ++cls) {
      if (logits[token * model->output_classes() + cls] >
          logits[token * model->output_classes() + expected]) {
        expected = cls;
      }
    }
    check(prediction.at("predicted").at(token).integer() == expected,
          "Inference prediction matches forward output");
  }
}

}  // namespace

int main() {
  try {
    json_tests();
    affine_tests();
    checkpoint_tests();
    inference_tests();
    std::cout << "Model/IO tests passed: " << checks << " checks\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Model/IO test failure after " << checks
              << " checks: " << error.what() << '\n';
    return 1;
  }
}
