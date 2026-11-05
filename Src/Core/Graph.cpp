#include "Core/Graph.h"

#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace elman {
namespace {

thread_local bool recording = true;

void require(bool condition, const std::string& message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

namespace graph {

RecordingGuard::RecordingGuard() : previous_(recording) {
  recording = false;
}

RecordingGuard::~RecordingGuard() {
  recording = previous_;
}

Tensor tracked(Tensor output, const std::vector<Tensor>& parents,
               std::function<void(const Tensor&)> backward) {
  if (!recording) {
    return output;
  }
  bool needed = false;
  for (const auto& parent : parents) {
    needed |= parent.requires_grad();
  }
  if (!needed) {
    return output;
  }
  output.node()->requires_grad = true;
  output.node()->backward = std::move(backward);
  output.node()->saved_version = output.node()->value->version;
  for (const auto& parent : parents) {
    if (parent.defined()) {
      output.node()->parents.push_back(parent.node());
      output.node()->versions.push_back(parent.node()->value->version);
    }
  }
  return output;
}

std::vector<std::shared_ptr<TensorNode>> topological_order(
    const Tensor& output) {
  std::vector<std::shared_ptr<TensorNode>> topology;
  std::unordered_set<TensorNode*> seen;
  std::function<void(const std::shared_ptr<TensorNode>&)> visit =
      [&](const std::shared_ptr<TensorNode>& node) {
        if (!seen.insert(node.get()).second) {
          return;
        }
        require(!node->backward || node->saved_version == node->value->version,
                "A graph output was modified before backward");
        for (size_t i = 0; i < node->parents.size(); ++i) {
          require(node->versions[i] == node->parents[i]->value->version,
                  "A graph input was modified before backward");
          visit(node->parents[i]);
        }
        topology.push_back(node);
      };
  visit(output.node());
  return topology;
}

}  // namespace graph

std::string Tensor::graph_json() const {
  require(defined(), "Undefined tensor");
  std::unordered_map<TensorNode*, size_t> ids;
  std::vector<std::shared_ptr<TensorNode>> nodes;
  std::function<void(const std::shared_ptr<TensorNode>&)> visit =
      [&](const std::shared_ptr<TensorNode>& node) {
        if (ids.count(node.get())) {
          return;
        }
        for (const auto& parent : node->parents) {
          visit(parent);
        }
        ids[node.get()] = nodes.size();
        nodes.push_back(node);
      };
  visit(node_);

  std::ostringstream stream;
  stream << "{\"format\":\"elman.graph\",\"version\":1,\"nodes\":[";
  for (size_t i = 0; i < nodes.size(); ++i) {
    const auto& node = nodes[i];
    if (i) {
      stream << ',';
    }
    stream << "{\"id\":" << i << ",\"op\":\"" << node->operation
           << "\",\"shape\":[";
    for (size_t j = 0; j < node->shape.size(); ++j) {
      if (j) {
        stream << ',';
      }
      stream << node->shape[j];
    }
    stream << "],\"requires_grad\":" << (node->requires_grad ? "true" : "false")
           << ",\"inputs\":[";
    for (size_t j = 0; j < node->parents.size(); ++j) {
      if (j) {
        stream << ',';
      }
      stream << ids[node->parents[j].get()];
    }
    stream << "]}";
  }
  stream << "],\"output\":" << ids[node_.get()] << '}';
  return stream.str();
}

}  // namespace elman
