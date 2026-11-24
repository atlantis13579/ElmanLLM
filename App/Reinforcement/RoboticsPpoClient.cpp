#include "RoboticsPpoClient.h"

#include "Core/Backend.h"
#include "Models/Model.h"
#include "Optimizer/Optimizer.h"
#include "Reinforcement/PpoRunner.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#else
#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace elman {
namespace {
constexpr std::array<int, 5> widths{86, 128, 128, 128, 25};
constexpr size_t weight_count = 47385;
#ifdef _WIN32
using NativeSocket = SOCKET;
constexpr NativeSocket invalid_socket = INVALID_SOCKET;
void close_socket(NativeSocket socket) { closesocket(socket); }
struct SocketRuntime {
  SocketRuntime() {
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data)) throw std::runtime_error("WSAStartup failed");
  }
  ~SocketRuntime() { WSACleanup(); }
};
#else
using NativeSocket = int;
constexpr NativeSocket invalid_socket = -1;
void close_socket(NativeSocket socket) { close(socket); }
struct SocketRuntime {};
#endif

struct Packet {
  char type = 0;
  std::vector<char> bytes;
};

class Channel {
 public:
  explicit Channel(int port) {
    if (port < 1 || port > 65535) throw std::invalid_argument("Invalid PPO port");
    socket_ = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket_ == invalid_socket) throw std::runtime_error("Cannot open PPO TCP socket");
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<uint16_t>(port));
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    for (int attempt = 0; attempt < 50; ++attempt) {
      if (connect(socket_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0) return;
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    throw std::runtime_error("Cannot connect to the Riemann PPO environment");
  }
  ~Channel() { if (socket_ != invalid_socket) close_socket(socket_); }
  Channel(const Channel&) = delete;

  void send_packet(char type, const void* data = nullptr, uint32_t size = 0) {
    if (size > 1024 * 1024) throw std::invalid_argument("PPO packet too large");
    const std::array<unsigned char, 5> header{
        static_cast<unsigned char>(type), static_cast<unsigned char>(size),
        static_cast<unsigned char>(size >> 8), static_cast<unsigned char>(size >> 16),
        static_cast<unsigned char>(size >> 24)};
    send_all(header.data(), header.size());
    if (size) send_all(data, size);
  }
  Packet receive() {
    std::array<unsigned char, 5> header{};
    receive_all(header.data(), header.size());
    const uint32_t size = uint32_t(header[1]) | (uint32_t(header[2]) << 8) |
                          (uint32_t(header[3]) << 16) | (uint32_t(header[4]) << 24);
    if (size > 1024 * 1024) throw std::runtime_error("PPO packet too large");
    Packet result;
    result.type = static_cast<char>(header[0]);
    result.bytes.resize(size);
    if (size) receive_all(result.bytes.data(), size);
    return result;
  }
  Packet request(char type, const void* data, uint32_t size, char expected) {
    std::lock_guard<std::mutex> lock(mutex_);
    send_packet(type, data, size);
    Packet reply = receive();
    if (reply.type == 'E') throw std::runtime_error(std::string(reply.bytes.begin(), reply.bytes.end()));
    if (reply.type != expected) throw std::runtime_error("Unexpected PPO reply");
    return reply;
  }

 private:
  void send_all(const void* data, size_t count) {
    const char* bytes = static_cast<const char*>(data);
    while (count) {
      const int sent = send(socket_, bytes, static_cast<int>(count), 0);
      if (sent <= 0) throw std::runtime_error("PPO connection closed while sending");
      bytes += sent;
      count -= static_cast<size_t>(sent);
    }
  }
  void receive_all(void* data, size_t count) {
    char* bytes = static_cast<char*>(data);
    while (count) {
      const int received = recv(socket_, bytes, static_cast<int>(count), 0);
      if (received <= 0) throw std::runtime_error("PPO connection closed while receiving");
      bytes += received;
      count -= static_cast<size_t>(received);
    }
  }
  NativeSocket socket_ = invalid_socket;
  std::mutex mutex_;
};

template <typename T> T decode(const Packet& packet, size_t offset = 0) {
  if (offset + sizeof(T) > packet.bytes.size()) throw std::runtime_error("Incomplete PPO packet");
  T value;
  std::memcpy(&value, packet.bytes.data() + offset, sizeof(value));
  return value;
}

std::vector<float> pack_weights(const Model& model) {
  const auto& parameters = model.parameters();
  if (parameters.size() != 8) throw std::runtime_error("Unexpected PPO model parameters");
  std::vector<float> packed;
  packed.reserve(weight_count);
  for (int layer = 0; layer < 4; ++layer) {
    const std::vector<float> weight = parameters[2 * layer].value.data();
    const std::vector<float> bias = parameters[2 * layer + 1].value.data();
    for (int out = 0; out < widths[layer + 1]; ++out)
      for (int in = 0; in < widths[layer]; ++in)
        packed.push_back(weight[in * widths[layer + 1] + out]);
    packed.insert(packed.end(), bias.begin(), bias.end());
  }
  if (packed.size() != weight_count) throw std::runtime_error("Unexpected PPO weight count");
  return packed;
}

void unpack_weights(Model& model, const Packet& packet) {
  if (packet.bytes.empty()) return;
  if (packet.bytes.size() != weight_count * sizeof(float))
    throw std::runtime_error("Invalid PPO initial weights");
  auto& parameters = model.parameters();
  size_t cursor = 0;
  for (int layer = 0; layer < 4; ++layer) {
    std::vector<float> weight(size_t(widths[layer]) * widths[layer + 1]);
    std::vector<float> bias(widths[layer + 1]);
    for (int out = 0; out < widths[layer + 1]; ++out)
      for (int in = 0; in < widths[layer]; ++in) {
        float value;
        std::memcpy(&value, packet.bytes.data() + cursor * sizeof(float), sizeof(float));
        weight[in * widths[layer + 1] + out] = value;
        ++cursor;
      }
    std::memcpy(bias.data(), packet.bytes.data() + cursor * sizeof(float), bias.size() * sizeof(float));
    cursor += bias.size();
    parameters[2 * layer].value.assign(weight);
    parameters[2 * layer + 1].value.assign(bias);
  }
}

std::vector<float> actor_forward(const std::vector<float>& weights, const std::vector<float>& input) {
  if (input.size() != 86) throw std::runtime_error("Invalid PPO observation width");
  std::vector<float> values = input;
  size_t cursor = 0;
  for (int layer = 0; layer < 4; ++layer) {
    std::vector<float> next(widths[layer + 1]);
    for (int out = 0; out < widths[layer + 1]; ++out) {
      float sum = 0;
      for (int in = 0; in < widths[layer]; ++in)
        sum += values[in] * weights[cursor + size_t(out) * widths[layer] + in];
      next[out] = sum + weights[cursor + size_t(widths[layer]) * widths[layer + 1] + out];
      if (layer != 3 && next[out] < 0) next[out] = std::exp(next[out]) - 1;
    }
    cursor += size_t(widths[layer] + 1) * widths[layer + 1];
    values = std::move(next);
  }
  return values;
}

class RemoteEnvironment final : public rl::PpoEnvironment {
 public:
  RemoteEnvironment(Channel& channel, uint8_t id) : channel_(channel), id_(id) {}
  std::vector<float> observe() const override {
    Packet reply = channel_.request('O', &id_, 1, 'o');
    if (reply.bytes.size() != 86 * sizeof(float)) throw std::runtime_error("Invalid PPO observation");
    std::vector<float> result(86);
    std::memcpy(result.data(), reply.bytes.data(), reply.bytes.size());
    return result;
  }
  rl::PpoOutcome step(const std::vector<float>& action) override {
    if (action.size() != 24) throw std::runtime_error("Invalid PPO action width");
    std::array<char, 1 + 24 * sizeof(float)> payload{};
    payload[0] = static_cast<char>(id_);
    std::memcpy(payload.data() + 1, action.data(), 24 * sizeof(float));
    Packet reply = channel_.request('S', payload.data(), payload.size(), 's');
    if (reply.bytes.size() != sizeof(float) + 1) throw std::runtime_error("Invalid PPO step result");
    return {decode<float>(reply), reply.bytes[sizeof(float)] != 0};
  }
  void reset() override { channel_.request('R', &id_, 1, 'r'); }

 private:
  Channel& channel_;
  uint8_t id_;
};
}  // namespace

int robotics_ppo_client(int port) {
  SocketRuntime runtime;
  Channel channel(port);
  try {
    const Packet config = channel.receive();
    const Packet initial = channel.receive();
    if (config.type != 'C' || config.bytes.size() != 3 * sizeof(uint32_t) || initial.type != 'I')
      throw std::runtime_error("Invalid PPO handshake");
    const uint32_t updates = decode<uint32_t>(config, 0);
    const uint32_t count = decode<uint32_t>(config, 4);
    const uint32_t rollout = decode<uint32_t>(config, 8);
    if (!updates || !count || count > 8 || rollout < 32 || rollout % count)
      throw std::runtime_error("Invalid PPO training settings");
    if (!backend_available(Backend::CUDA)) throw std::runtime_error("Robotics PPO needs CUDA");
    const Json definition = Json::parse(R"({"format":"elman.model","version":1,
      "architecture":"continuous_actor_critic","name":"robotics","input_features":86,
      "actions":24,"hidden_features":128,"hidden_layers":3})");
    std::unique_ptr<Model> model = create_model(definition, Backend::CUDA, 42);
    if (model->parameters().size() != 8 || model->parameters()[6].name != "layers.3.weight" ||
        model->parameters()[6].value.shape() != Shape{128, 25})
      throw std::runtime_error("Unexpected robotics PPO model layout");
    model->parameters()[6].value.assign(std::vector<float>(model->parameters()[6].value.numel(), 0));
    const std::vector<float> baseline = pack_weights(*model);
    channel.send_packet('B', baseline.data(), static_cast<uint32_t>(baseline.size() * sizeof(float)));
    unpack_weights(*model, initial);
    const std::vector<float> current = pack_weights(*model);
    channel.send_packet('W', current.data(), static_cast<uint32_t>(current.size() * sizeof(float)));
    if (channel.receive().type != 'A') throw std::runtime_error("PPO handshake rejected");
    Optimizer optimizer(Json::Object{{"optimizer", "adam"}, {"learning_rate", 0.00001},
                                     {"clip_norm", 1.0}}, *model, true);
    std::vector<std::unique_ptr<RemoteEnvironment>> owned;
    std::vector<rl::PpoEnvironment*> environments;
    for (uint32_t id = 0; id < count; ++id) {
      owned.emplace_back(std::make_unique<RemoteEnvironment>(channel, static_cast<uint8_t>(id)));
      environments.push_back(owned.back().get());
      owned.back()->reset();
    }
    rl::PpoRunnerSettings settings;
    settings.rollout_actions = static_cast<int>(rollout);
    settings.standard_deviation = 0.15f;
    settings.reward_scale = 0.02f;
    rl::PpoRunner runner(*model, optimizer, std::move(environments),
      [](const Model& source) {
        auto weights = std::make_shared<std::vector<float>>(pack_weights(source));
        return rl::PpoInference([weights](const std::vector<float>& observation) {
          return actor_forward(*weights, observation);
        });
      }, settings);
    runner.run(static_cast<int>(updates), [&](const rl::PpoIteration& iteration) {
      const std::vector<float> weights = pack_weights(*model);
      std::vector<char> payload(3 * sizeof(uint32_t) + 3 * sizeof(double) + weights.size() * sizeof(float));
      const uint32_t update = static_cast<uint32_t>(iteration.update);
      const uint32_t episodes = static_cast<uint32_t>(iteration.episodes);
      std::memcpy(payload.data(), &update, sizeof(update));
      std::memcpy(payload.data() + 4, &iteration.mean_reward, sizeof(float));
      std::memcpy(payload.data() + 8, &iteration.policy_loss, sizeof(double));
      std::memcpy(payload.data() + 16, &iteration.value_loss, sizeof(double));
      std::memcpy(payload.data() + 24, &iteration.gradient_norm, sizeof(double));
      std::memcpy(payload.data() + 32, &episodes, sizeof(episodes));
      std::memcpy(payload.data() + 36, weights.data(), weights.size() * sizeof(float));
      channel.request('U', payload.data(), static_cast<uint32_t>(payload.size()), 'u');
    });
    channel.send_packet('D');
    return 0;
  } catch (const std::exception& error) {
    const std::string message = error.what();
    try { channel.send_packet('E', message.data(), static_cast<uint32_t>(message.size())); } catch (...) {}
    throw;
  }
}
}  // namespace elman
