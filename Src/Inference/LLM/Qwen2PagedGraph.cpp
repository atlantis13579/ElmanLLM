#include "Inference/LLM/Qwen2PagedGraph.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "Core/Device.h"
#include "Inference/LLM/Qwen2Graph.h"

namespace elman {
namespace {

void require(bool valid, const char* message) { if (!valid) throw std::invalid_argument(message); }

void validate_tokens(const Qwen2& model, int batch, const std::vector<int32_t>& tokens) {
  require(tokens.size() == size_t(batch), "Paged decode graph token count differs");
  for (int32_t token : tokens)
    require(token >= 0 && token < model.config().vocabulary, "Paged decode graph token exceeds the vocabulary");
}

}  // namespace

Qwen2PagedGraph::Qwen2PagedGraph(const Qwen2& model, PagedKVBatch& batch, int capacity,
                                 const std::vector<int32_t>& tokens)
    : model_(model), batch_(batch.batch()), capacity_(capacity) {
  require(model.backend() == Backend::CUDA && model.supports_paged_cache() && model.tiled_attention_enabled() &&
          current_device(Backend::CUDA) == model.input_device(), "Paged decode graph requires a single CUDA Qwen decoder");
  require(capacity_ >= 1 && capacity_ <= model.config().max_positions, "Invalid paged decode graph context capacity");
  validate_tokens(model_, batch_, tokens);
  auto append = batch.append(1);
  auto bindings = append->prepare_graph(capacity_);
  std::vector<int32_t> visible(bindings.slots.size());
  for (size_t i = 0; i < visible.size(); ++i) visible[i] = bindings.slots[i] >= 0;
  map_ = std::make_unique<PagedAttentionMap>(batch_, bindings.physical_slots, bindings.slots,
                                            visible, Backend::CUDA, false);
  destinations_ = Tensor::from_int_data({batch_}, bindings.destinations, Backend::CUDA);
  storage_ = std::move(bindings.layers);
  const auto input = Tensor::from_int_data({batch_, 1}, tokens);
  const auto positions = Tensor::from_int_data({batch_, 1}, bindings.positions);
  const Qwen2::PagedGraphInputs page_bindings{*map_, destinations_, storage_};
  graph_.record([&] {
    logits_ = model_.forward_impl(input, {}, positions, nullptr, nullptr, false, DecoderLogits::Last, {},
                                  &inputs_, &page_bindings).logits;
  });
  graph_.compile();
  append->complete_graph();
}

Tensor Qwen2PagedGraph::decode(PagedKVBatch& batch, const std::vector<int32_t>& tokens) {
  require(batch.batch() == batch_, "Paged decode graph batch width differs");
  validate_tokens(model_, batch_, tokens);
  DeviceScope owner(Backend::CUDA, model_.input_device());
  auto append = batch.append(1);
  const auto bindings = append->prepare_graph(capacity_);
  require(bindings.layers.size() == storage_.size(), "Paged decode graph layer storage differs");
  for (size_t i = 0; i < storage_.size(); ++i) {
    require(bindings.layers[i].key.node() == storage_[i].key.node() &&
            bindings.layers[i].value.node() == storage_[i].value.node(), "Paged decode graph cache pool changed");
  }
  inputs_.indices.assign_int(tokens);
  const auto rotary = model_.rotary_values(bindings.positions);
  inputs_.cosine.assign(rotary.first);
  inputs_.sine.assign(rotary.second);
  destinations_.assign_int(bindings.destinations);
  map_->update_graph_slots(bindings.slots);
  graph_.replay();
  graph_.synchronize();
  append->complete_graph();
  return logits_;
}

int Qwen2PagedGraph::context_bucket(int required, int maximum) {
  require(required > 0 && required <= maximum, "Paged decode graph context exceeds the worker maximum");
  return static_cast<int>(std::min<int64_t>(maximum, ((int64_t(required) + 63) / 64) * 64));
}

size_t Qwen2PagedGraph::memory_bound(const Qwen2& model, int batch, int capacity) {
  require(batch > 0 && capacity > 0, "Invalid paged decode graph memory geometry");
  return checked_memory_mul(batch, Qwen2Graph::memory_bound(model.config(), model.dtype(), std::max(2, capacity),
                                                              model.quantization_workspace_bytes()));
}

}  // namespace elman
