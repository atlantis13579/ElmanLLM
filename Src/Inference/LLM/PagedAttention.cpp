#include "Inference/LLM/PagedAttention.h"
#include "Core/TensorDispatch.h"
#include "Core/Graph.h"
#include "Core/Memory.h"
#include "Operators/paged_attention.h"
#include <climits>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace elman {
namespace {
void require(bool value, const char* message) {
    if (!value) throw std::invalid_argument(std::string("Paged attention: ") + message);
}
struct MapValidation {
    std::weak_ptr<Buffer> buffer;
    uint64_t version = 0;
    int slots = -1;
    bool contiguous = false;
    int length = -1;
};
thread_local MapValidation validated_map;
void remember(const BufferPtr& buffer, int slots, bool contiguous, int length) { validated_map = {buffer, buffer->version, slots, contiguous, length}; }
bool validate_map(const BufferPtr& buffer, int slots, int length) {
    if (validated_map.buffer.lock() == buffer && validated_map.version == buffer->version && validated_map.slots == slots && validated_map.length == length)
        return validated_map.contiguous;
    bool contiguous = length > 0;
    int first = 0, index = 0;
    for (int32_t value : backend_download_int(buffer)) {
        if (value == INT_MIN) { contiguous = false; ++index; continue; }
        const int slot = value < 0 ? -value - 1 : value;
        require(slot < slots, "physical slot is outside storage");
        if (length && index % length == 0) first = slot;
        contiguous &= length > 0 && int64_t(slot) == int64_t(first) + index % length;
        ++index;
    }
    remember(buffer, slots, contiguous, length);
    return contiguous;
}
}
PagedAttentionMap::PagedAttentionMap(int batch, int physical_slots, const std::vector<int32_t>& slots,
                                   const std::vector<int32_t>& visible, Backend backend, bool allow_contiguous)
    : batch_(batch), length_(0), physical_slots_(physical_slots) {
    require(batch > 0 && physical_slots >= 0 && slots.size() <= size_t(INT_MAX) && slots.size() % batch == 0,
            "invalid map dimensions");
    require(visible.size() == slots.size(), "mask shape differs");
    length_ = int(slots.size() / batch);
    MemoryReservation temporary(Backend::CPU, checked_memory_mul(slots.size(), sizeof(int32_t)));
    std::vector<int32_t> encoded(slots.size());
    contiguous_ = allow_contiguous && length_ > 0;
    for (size_t i = 0; i < slots.size(); ++i) {
        require(slots[i] >= -1 && slots[i] < physical_slots, "invalid physical slot");
        require(visible[i] == 0 || visible[i] == 1, "mask must contain zero or one");
        require(slots[i] >= 0 || !visible[i], "structural padding cannot be exposed");
        encoded[i] = slots[i] < 0 ? INT_MIN : visible[i] ? slots[i] : -slots[i] - 1;
        contiguous_ &= slots[i] >= 0 && int64_t(slots[i]) == int64_t(slots[(i / length_) * length_]) + i % length_;
    }
    encoded_ = Tensor::from_int_data({batch_, length_}, encoded, backend);
    remember(encoded_.node()->value, physical_slots_, contiguous_, length_);
}
void PagedAttentionMap::update_graph_slots(const std::vector<int32_t>& slots) {
    require(!contiguous_ && slots.size() == encoded_.numel(), "graph paged map geometry changed");
    MemoryReservation temporary(Backend::CPU, checked_memory_mul(slots.size(), sizeof(int32_t)));
    std::vector<int32_t> encoded;
    encoded.reserve(slots.size());
    for (int32_t slot : slots) {
        require(slot >= -1 && slot < physical_slots_, "graph paged map has an invalid slot");
        encoded.push_back(slot < 0 ? INT_MIN : slot);
    }
    encoded_.assign_int(encoded);
}
Tensor paged_attention(const Tensor& query, const Tensor& keys, const Tensor& values,
                       const PagedAttentionMap& map, int kv_heads, float divisor,
                       int query_tile, bool causal, int causal_offset) {
    require(query.defined() && keys.defined() && values.defined(), "undefined operand");
    require(!query.requires_grad() && !keys.requires_grad() && !values.requires_grad(), "inference operands cannot require gradients");
    require(keys.shape().size() == 2 && keys.shape()[0] == map.physical_slots_, "map storage differs");
    int bits;
    std::memcpy(&bits, &divisor, sizeof(bits));
    return dispatch_tensor("paged_attention", query.shape(), {query, keys, values, map.encoded_},
                           {bits, query_tile, int(causal), causal_offset, kv_heads, int(map.contiguous_)}, query.dtype());
}
void validate_paged_attention(const BufferPtr& output, const std::vector<BufferPtr>& inputs, const KernelParams& p) {
    require(bool(output) && inputs.size() == 4, "four operands are required");
    require(is_compute_float(output->dtype) && p.output_dtype() == int(output->dtype), "unsupported output dtype");
    for (int i = 0; i < 4; ++i) {
        const auto id = KernelInput(i);
        const auto type = i == 3 ? DType::Int32 : output->dtype;
        require(inputs[i] && inputs[i]->backend == output->backend && inputs[i]->dtype == type && p.input_dtype(id) == int(type),
                "operand backend or dtype differs");
        const int rank = i == 0 ? 4 : 2;
        require(p.input_rank(id) == rank, "operand rank differs");
        size_t count = 1;
        for (int d = rank - 1; d >= 0; --d) {
            const int length = p.input_shape(id, d);
            require(length >= 0 && count <= size_t(INT_MAX) && p.input_stride(id, d) == int(count), "invalid contiguous descriptor");
            require(!length || count <= size_t(INT_MAX) / length, "operand indexing overflows");
            count *= size_t(length);
        }
        require(count == inputs[i]->count, "operand storage size differs");
        if (inputs[i]->bytes && output->bytes) {
            const auto a = reinterpret_cast<uintptr_t>(output->raw_pointer()), b = reinterpret_cast<uintptr_t>(inputs[i]->raw_pointer());
            require(a <= b ? b - a >= output->bytes : a - b >= inputs[i]->bytes, "output overlaps an operand");
        }
    }
    PagedAttentionShape s(p);
    require(s.batch > 0 && s.heads > 0 && s.dimension > 0 && s.kv_heads > 0 && s.heads % s.kv_heads == 0, "invalid head geometry");
    require(p.input_shape(KernelInput::B, 1) == int64_t(s.kv_heads) * s.dimension &&
            p.input_shape(KernelInput::C, 0) == s.slots && p.input_shape(KernelInput::C, 1) == p.input_shape(KernelInput::B, 1),
            "physical K/V geometry differs");
    require(p.input_shape(KernelInput::D, 0) == s.batch, "map batch differs");
    require(p.output_rank() == 4 && p.output_count() == int(output->count) && output->count == inputs[0]->count,
            "output shape or count differs");
    for (int d = 0; d < 4; ++d) require(p.output_shape(d) == p.input_shape(KernelInput::A, d), "output shape differs");
    float divisor;
    const int bits = p.attribute(0);
    std::memcpy(&divisor, &bits, sizeof(divisor));
    require(std::isfinite(divisor) && divisor > 0 && p.attribute(1) > 0 && p.attribute(1) <= 1024,
            "invalid scale or query tile");
    require((p.attribute(2) == 0 || p.attribute(2) == 1) && p.attribute(3) >= 0, "invalid causal descriptor");
    require(int64_t(s.batch) * s.heads * s.tile * s.keys <= INT_MAX, "score tile exceeds indexing limit");
    require(p.attribute(5) == 0 || p.attribute(5) == 1, "invalid contiguous map flag");
    const bool contiguous = validate_map(inputs[3], s.slots, s.keys);
    require(!p.attribute(5) || contiguous, "contiguous flag disagrees with physical map");
}
} // namespace elman
