#include "PolicyValueServer.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <list>
#include <limits>
#include <mutex>
#include <sstream>

#include "IO/AtomicFile.h"
#include "IO/Checkpoint.h"
#include "Messages.h"
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>
#endif

namespace elman {
namespace {
using Path = std::filesystem::path;

// One writable model per run directory. Kernel ownership is released on process
// exit.
class RunLock {
 public:
  explicit RunLock(const Path& dir) {
    if (dir.empty()) {
      return;
    }
    std::filesystem::create_directories(dir);
    std::filesystem::path path = dir / "training.lock";
#ifdef _WIN32
    handle_ = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr, OPEN_ALWAYS,
                          FILE_ATTRIBUTE_NORMAL, nullptr);
    if (handle_ == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("Cannot exclusively open training directory: " +
                               dir.u8string());
    }
#else
    handle_ = open(path.c_str(), O_CREAT | O_RDWR, 0600);
    if (handle_ == -1 || flock(handle_, LOCK_EX | LOCK_NB) != 0) {
      if (handle_ >= 0) {
        close(handle_);
      }
      throw std::runtime_error("Cannot exclusively open training directory: " +
                               dir.u8string());
    }
#endif
  }
  ~RunLock() {
#ifdef _WIN32
    if (handle_ != INVALID_HANDLE_VALUE) {
      CloseHandle(handle_);
    }
#else
    if (handle_ >= 0) {
      close(handle_);
    }
#endif
  }
  RunLock(const RunLock&) = delete;

 private:
#ifdef _WIN32
  HANDLE handle_ = INVALID_HANDLE_VALUE;
#else
  int handle_ = -1;
#endif
};

std::string legacy_filename(int revision) {
  std::ostringstream s;
  s << "update-" << std::setw(8) << std::setfill('0') << revision << ".elman";
  return s.str();
}

void write_json(const Path& path, const Json& value) {
  write_file_atomically(path, [&](const Path& tmp) { value.write(tmp.u8string()); });
}

class PolicyValueService {
 public:
  PolicyValueService(const Path& config, const Path& initial, Path directory,
                     Backend backend, const Json& settings,
                     std::string model_type)
      : directory_(std::move(directory)),
        lock_(directory_),
        definition_(Json::read(config)),
        backend_(backend),
        model_type_(std::move(model_type)),
        model_(create_model(definition_, backend, settings.int_or("seed", 42))),
        optimizer_(Json::Object{{"optimizer", "adam"},
                                {"learning_rate",
                                 settings.number_or("learning_rate", .001)},
                                {"weight_decay", settings.number_or("weight_decay", 0)},
                                {"clip_grad_norm", settings.number_or("clip_grad_norm", 1)}},
                   *model_) {
    if (model_->architecture() != "spatial_policy_value" &&
        model_->architecture() != "spatial_policy_value_dense") {
      throw std::invalid_argument(
          "Policy-value serving needs a spatial_policy_value definition");
    }
    Path path = initial;
    Json manifest;
    if (!directory_.empty()) {
      const Path index = directory_ / "latest.json";
      if (std::filesystem::exists(index)) {
        manifest = Json::read(index);
        const std::string name = manifest.at("checkpoint").string();
        if (name != "latest.elman" && name != legacy_filename(manifest.at("revision").integer()))
          throw std::runtime_error("Invalid checkpoint filename");
        path = directory_ / name;
      }
      if (std::filesystem::exists(directory_ / "latest.elman"))
        path = directory_ / "latest.elman";
    }
    if (!path.empty()) {
      load_checkpoint(path.u8string(), *model_, definition_,
                      &optimizer_.state());
    } else if (directory_.empty()) {
      throw std::invalid_argument("Inference requires a checkpoint");
    }
    if (!directory_.empty() && path == directory_ / "latest.elman") {
      // The weights and index are separate files. Match the candidate index to
      // the optimizer step actually present in the atomically replaced weights.
      const Path pending = directory_ / "pending.json";
      if (std::filesystem::exists(pending)) {
        const Json candidate = Json::read(pending);
        if (candidate.at("step").number() == double(optimizer_.state().step)) {
          manifest = candidate;
          write_json(directory_ / "latest.json", manifest);
        }
        std::filesystem::remove(pending);
      }
      if (manifest.is_null() || manifest.at("checkpoint").string() != "latest.elman")
        throw std::runtime_error("Missing latest checkpoint metadata");
    }
    if (!manifest.is_null()) {
      revision_ = saved_ = manifest.at("revision").integer();
      progress_ = manifest.int_or("progress", -1);
      saved_name_ = manifest.at("checkpoint").string();
      if (revision_ < 0 || manifest.at("step").number() != double(optimizer_.state().step))
        throw std::runtime_error("Inconsistent checkpoint metadata");
    }
  }

  Json handle(const Json& body) {
    if (body.at("version").integer() != 1) {
      throw std::invalid_argument("Unsupported protocol version");
    }
    std::string type = body.at("type").string();
    bool predict = type == "predict", train = type == "train";
    std::lock_guard<std::mutex> guard(mutex_);
    if (type == "status") {
      return state();
    }
    if (!predict && !train && type != "checkpoint" && type != "archive") {
      throw std::invalid_argument("Unknown policy-value message");
    }
    if (!predict && directory_.empty()) {
      throw std::invalid_argument("Training is disabled");
    }
    const bool frozen = body.contains("snapshot");
    if (frozen && (!predict || directory_.empty()))
      throw std::invalid_argument("Snapshot selection requires prediction on a writable service");
    const int revision = predict && !frozen ? body.int_or("revision", revision_) : body.at("revision").integer();
    if (!frozen && revision != revision_)
      throw std::invalid_argument("Model revision changed; read status");
    if (predict) {
      const int progress = body.int_or("snapshot", -1);
      Model& selected = frozen ? snapshot_model(revision, progress) : *model_;
      Json::Object reply = policy::prediction_message(
          policy::predict(selected, policy::read_observation(body)));
      reply["revision"] = revision;
      if (frozen) reply["snapshot"] = progress;
      return reply;
    }
    if (body.int_or("expected_progress", progress_) != progress_)
      throw std::invalid_argument("Saved client progress changed; read status");
    if (train) {
      return update(body);
    }
    if (type == "archive") {
      if (revision_ != saved_) throw std::invalid_argument("Checkpoint the current revision before archiving");
      const Path name = snapshot_name(revision_, progress_), path = directory_ / name;
      if (!std::filesystem::exists(path)) {
        if (saved_name_.empty())
          save_checkpoint(path.u8string(), *model_, definition_, optimizer_.state());
        else
          write_file_atomically(path, [&](const Path& tmp) { std::filesystem::copy_file(directory_ / saved_name_, tmp); });
      }
      return Json::Object{{"revision", revision_}, {"snapshot", progress_}, {"checkpoint", name.generic_string()}};
    }
    const int progress = body.int_or("progress", progress_);
    if (progress < progress_) throw std::invalid_argument("Checkpoint progress moved backwards");
    if (saved_name_ != "latest.elman" || revision_ != saved_ || progress != progress_) {
      const std::string name = "latest.elman";
      Json manifest = Json::Object{{"checkpoint", name},
                                   {"revision", revision_},
                                   {"progress", progress},
                                   {"step", double(optimizer_.state().step)}};
      write_json(directory_ / "pending.json", manifest);
      if (saved_name_ != name || revision_ != saved_)
        save_checkpoint((directory_ / name).u8string(), *model_, definition_, optimizer_.state());
      write_json(directory_ / "latest.json", manifest);
      saved_ = revision_;
      progress_ = progress;
      saved_name_ = name;
      std::filesystem::remove(directory_ / "pending.json");
    }
    return state();
  }

 private:
  static Path snapshot_name(int revision, int progress) {
    if (revision < 0 || progress < -1) throw std::invalid_argument("Invalid snapshot identifier");
    return Path("backups") / ("revision-" + std::to_string(revision) + "-progress-" + std::to_string(progress) + ".elman");
  }
  Model& snapshot_model(int revision, int progress) {
    const Path path = directory_ / snapshot_name(revision, progress);
    auto found = std::find_if(snapshots_.begin(), snapshots_.end(), [&](const auto& item) { return item.first == path; });
    if (found == snapshots_.end()) {
      if (!std::filesystem::is_regular_file(path)) throw std::invalid_argument("Unknown snapshot");
      auto frozen = create_model(definition_, backend_, 0);
      load_checkpoint(path.u8string(), *frozen, definition_);
      snapshots_.emplace_front(path, std::move(frozen));
      if (snapshots_.size() > 2) snapshots_.pop_back();
    } else {
      snapshots_.splice(snapshots_.begin(), snapshots_, found);
    }
    return *snapshots_.front().second;
  }
  Json::Object state() const {
    return {{"version", 1},
            {"status", "ok"},
            {"model_type", model_type_},
            {"model", model_->name()},
            {"backend", backend_name(backend_)},
            {"training", !directory_.empty()},
            {"revision", revision_},
            {"step", double(optimizer_.state().step)},
            {"saved_revision", saved_},
            {"progress", progress_},
            {"checkpoint", saved_name_}};
  }
  Json update(const Json& body) {
    if (revision_ == std::numeric_limits<int>::max())
      throw std::overflow_error("Policy-value revision limit reached");
    const policy::Objective objective = policy::read_objective(body);
    const int batch = body.at("batch_size").integer();
    const int seed = body.at("seed").integer();
    if (seed < 0 || batch < 1 || batch > 1024) {
      throw std::invalid_argument("Invalid batch size or seed");
    }
    const Json::Array& rows = body.at("samples").array();
    if (rows.empty() || rows.size() > 65536) {
      throw std::invalid_argument("Invalid sample count");
    }
    std::vector<policy::Sample> samples;
    samples.reserve(rows.size());
    for (const Json& row : rows) {
      samples.push_back(policy::read_sample(row));
    }
    std::mt19937 rng(seed);
    policy::UpdateResult update = policy::train_batch(*model_, optimizer_, samples, batch,
                                      objective, rng, body.bool_or("preselected", false));
    ++revision_;
    Json::Object result = state();
    result["loss"] = policy::loss_message(update);
    result["samples"] = static_cast<int>(samples.size());
    return result;
  }
  Path directory_;
  RunLock lock_;
  Json definition_;
  Backend backend_;
  std::string model_type_;
  std::unique_ptr<Model> model_;
  Optimizer optimizer_;
  std::mutex mutex_;
  std::string saved_name_;
  std::list<std::pair<Path, std::unique_ptr<Model>>> snapshots_;
  int revision_ = 0, saved_ = 0, progress_ = -1;
};
}  // namespace

void serve_policy_value(const Json& config, const std::filesystem::path& base,
                        Backend backend, const std::function<bool()>& stop) {
  const Json& model = config.at("model");
  const Json train = config.contains("training") ? config.at("training")
                                                 : Json(Json::Object{});
  LocalFunction relative{[&](const std::string& s) {
    return s.empty() ? Path{} : base / std::filesystem::u8path(s);
  }};
  PolicyValueService service(relative(model.at("path").string()),
                             relative(model.string_or("checkpoint", "")),
                             relative(train.string_or("directory", "")),
                             backend, train, model.at("type").string());
  const Json tcp =
      config.contains("tcp") ? config.at("tcp") : Json(Json::Object{});
  TcpOptions options;
  options.address = tcp.string_or("host", "127.0.0.1");
  options.port = tcp.int_or("port", 8091);
  options.max_connections = tcp.int_or("max_connections", 32);
  options.read_timeout_ms = tcp.int_or("read_timeout_ms", 30000);
  options.write_timeout_ms = tcp.int_or("write_timeout_ms", 30000);
  int limit = tcp.int_or("max_message_bytes", 32 * 1024 * 1024);
  if (limit <= 0) {
    throw std::invalid_argument("max_message_bytes must be positive");
  }
  using Clock = std::chrono::steady_clock;
  ConnectionCounts counts;
  listen_tcp(
      options, "tcp",
      "{\"version\":1,\"id\":0,\"type\":\"error\",\"error\":\"Server busy\"}\n",
      stop,
      [&](const TcpStream& connection) {
        std::string pending;
        for (;;) {
          size_t end = pending.find('\n');
          std::chrono::steady_clock::time_point deadline =
              Clock::now() + std::chrono::milliseconds(options.read_timeout_ms);
          while (end == pending.npos) {
            char bytes[8192];
            const uint64_t n = connection.read(
                bytes, sizeof(bytes),
                pending.empty() ? Clock::time_point::max() : deadline);
            if (pending.empty()) {
              deadline = Clock::now() +
                         std::chrono::milliseconds(options.read_timeout_ms);
            }
            size_t offset = pending.size();
            pending.append(bytes, n);
            end = pending.find('\n', offset);
            if ((end == pending.npos ? pending.size() : end) >
                static_cast<size_t>(limit)) {
              throw std::runtime_error(
                  "Policy-value message exceeds configured size");
            }
          }
          int id = 0;
          Json::Object reply;
          try {
            Json request = Json::parse(pending.substr(0, end));
            id = request.at("id").integer();
            std::string type = request.at("type").string();
            reply = service.handle(request).object();
            reply["type"] = type == "predict" ? "prediction" : type;
          } catch (const std::exception& e) {
            reply = {{"type", "error"}, {"error", e.what()}};
          }
          reply["version"] = 1;
          reply["id"] = id;
          connection.send(Json(std::move(reply)).dump(0) + '\n');
          pending.erase(0, end + 1);
        }
      },
      counts);
}
}  // namespace elman
