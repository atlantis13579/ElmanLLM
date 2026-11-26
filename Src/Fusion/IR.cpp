#include "Fusion/IR.h"
#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace elman::fusion {
namespace {
Dimensions broadcast(const Dimensions& a, const Dimensions& b) {
    Dimensions result(std::max(a.size(), b.size()), 1);
    for (size_t i = 0; i < result.size(); ++i) {
        const int x = i < a.size() ? a[a.size() - i - 1] : 1;
        const int y = i < b.size() ? b[b.size() - i - 1] : 1;
        if (x != y && x != 1 && y != 1) throw std::invalid_argument("Fusion broadcast mismatch");
        result[result.size() - i - 1] = std::max(x, y);
    }
    return result;
}
int append(Graph& g, Value v) { g.values.push_back(std::move(v)); return int(g.values.size()) - 1; }
}
size_t elements(const Dimensions& shape) { size_t n = 1; for (int d : shape) n *= size_t(d); return n; }
bool broadcasts_to(const Dimensions& a, const Dimensions& b) {
    if (a.size() > b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (a[i] != 1 && a[i] != b[b.size() - a.size() + i]) return false;
    return true;
}
bool is_reduction(Op op) {
    return op == Op::Sum || op == Op::Dot || op == Op::Maximum || op == Op::Mean || op == Op::MomentsMean || op == Op::MomentsVariance;
}
int Graph::input(const BufferPtr& buffer, const Dimensions& shape) {
    // A differently shaped or offset view is a region boundary, not an implicit reinterpretation.
    for (const Store& s : stores) if (bindings[s.binding]->raw_pointer() == buffer->raw_pointer() &&
            bindings[s.binding]->count == buffer->count) {
        if (values[s.value].shape != shape) throw std::invalid_argument("Fusion view boundary");
        return s.value;
    }
    for (size_t i = 0; i < values.size(); ++i) if (values[i].op == Op::Input && values[i].shape == shape &&
            bindings[values[i].slot] == buffer) return int(i);
    const int binding = int(bindings.size()); bindings.push_back(buffer);
    return append(*this, {Op::Input, shape, {}, 0, binding});
}
int Graph::constant(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    for (size_t i = 0; i < values.size(); ++i) if (values[i].op == Op::Constant && values[i].bits == bits) return int(i);
    return append(*this, {Op::Constant, {}, {}, bits});
}
int Graph::parameter(uint32_t bits) {
    const int slot = int(parameters.size()); parameters.push_back(bits);
    return append(*this, {Op::Parameter, {}, {}, 0, slot});
}
int Graph::scalar(float value) { uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); return parameter(bits); }
int Graph::unary(Op op, int x) { return append(*this, {op, values.at(x).shape, {x}}); }
int Graph::binary(Op op, int x, int y) { return append(*this, {op, broadcast(values.at(x).shape, values.at(y).shape), {x, y}}); }
int Graph::ternary(Op op, int x, int y, int z) {
    return append(*this, {op, broadcast(broadcast(values.at(x).shape, values.at(y).shape), values.at(z).shape), {x, y, z}});
}
int Graph::reduce(Op op, int x, ReductionOrder order) {
    Dimensions shape = values.at(x).shape;
    if (shape.empty()) throw std::invalid_argument("Fusion reduction requires an axis");
    shape.back() = 1;
    return append(*this, {op, shape, {x}, 0, int(order)});
}
int Graph::uniform(const Dimensions& shape, uint32_t seed, uint32_t offset) {
    const int first = parameter(seed), second = parameter(offset);
    return append(*this, {Op::Uniform, shape, {first, second}});
}
int Graph::dot(int x, int y) {
    Dimensions shape=broadcast(values.at(x).shape,values.at(y).shape);
    if(shape.empty()) throw std::invalid_argument("Fusion dot reduction requires an axis");
    shape.back()=1;
    return append(*this,{Op::Dot,shape,{x,y},0,int(ReductionOrder::Warp32)});
}
void Graph::store(int value, const BufferPtr& buffer) {
    const int binding = int(bindings.size()); bindings.push_back(buffer);
    stores.push_back({value, binding});
}
bool analyze(const Graph& graph, Schedule& result) {
    result = {};
    if (graph.values.empty() || graph.values.size() > 160 || graph.bindings.size() > 96 || graph.parameters.size() > 96) return false;
    for (const Value& v : graph.values) {
        if (elements(v.shape) > elements(result.domain) || v.shape.size() > result.domain.size()) result.domain = v.shape;
        result.reduction |= is_reduction(v.op);
        result.vectorized_lanes |= is_reduction(v.op) && v.slot==int(ReductionOrder::Vector4Warp32);
    }
    const size_t count = elements(result.domain);
    // Padded rows and lanes also compute indices, even when their stores are masked.
    if (!count || count > size_t(0x7fffffff - 4096)) return false;
    result.width = result.reduction ? result.domain.back() : 1;
    result.rows = int(count) / result.width;
    if (result.reduction && (result.width <= 0 || result.width > 1024)) return false;
    // Independent pointwise elements can share a thread without reordering any
    // arithmetic. Larger domains amortize scheduling over four coalesced passes.
    result.items = result.reduction ? (result.width + 31) / 32 : count >= 4096 ? 4 : 1;
    if(result.vectorized_lanes) result.items=((result.width+127)/128)*4;
    Dimensions row = result.domain;
    if (!row.empty()) row.back() = 1;
    for (const Value& v : graph.values) {
        // Moment trees have data-dependent overflow/cancellation behavior. Until
        // their original traversal is represented, leave them at the eager boundary.
        if (v.op == Op::MomentsMean || v.op == Op::MomentsVariance) return false;
        if(result.vectorized_lanes && is_reduction(v.op) && v.op!=Op::Maximum && v.slot!=int(ReductionOrder::Vector4Warp32)) return false;
        if (!broadcasts_to(v.shape, result.domain)) return false;
        if (v.op == Op::Input || v.op == Op::Constant || v.op == Op::Parameter) continue;
        if (v.shape != result.domain && (!result.reduction || v.shape != row)) return false;
        if (is_reduction(v.op) && (v.shape != row || graph.values.at(v.inputs.at(0)).shape != result.domain)) return false;
    }
    for (const Store& s : graph.stores) {
        const auto& shape = graph.values.at(s.value).shape;
        if (shape != result.domain && (!result.reduction || shape != row)) return false;
        if (elements(shape) != graph.bindings[s.binding]->count) return false;
    }
    return true;
}
} // namespace elman::fusion
