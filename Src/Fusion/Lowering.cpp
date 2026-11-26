#include "Fusion/IR.h"
#include "Kernals/float64_random_math.h"
#include "Kernals/low_precision_attention.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_set>

namespace elman::fusion {
namespace {
float from_bits(int bits) { float result; std::memcpy(&result, &bits, sizeof(result)); return result; }
Dimensions input_shape(const KernelParams& p, int index) {
    Dimensions s;
    for (int d = 0; d < p.input_rank(KernelInput(index)); ++d) s.push_back(p.input_shape(KernelInput(index), d));
    return s;
}
bool last_axis(const KernelParams& p) { return p.input_rank(KernelInput::A) > 0 && p.attribute(0) == p.input_rank(KernelInput::A) - 1; }
}
// These are mathematical operator definitions. The analyzer never matches operator
// names or model subgraphs: it only sees the resulting primitive dataflow.
bool has_definition(const std::string& operation) {
    static const std::unordered_set<std::string> definitions{
        "add","sub","mul","div","pow","exp","sqrt","rsqrt","neg","sum","mean","max",
        "gelu","softmax","attention_softmax","dropout"};
    return definitions.count(operation)!=0;
}
bool lower(Graph& g, const Command& c) {
    const auto& p = c.parameters;
    if (c.output->dtype != DType::Float32 || !c.output->count || p.output_rank() < 0 || p.output_rank() > 8) return false;
    // Low-precision attention reuses the attention_softmax name for tagged FP32
    // probabilities, derivatives, matrix products and reductions. Their input
    // arity and attributes differ from scaled/masked softmax below (probabilities
    // have no mask at all). Keep these dispatches on their dedicated kernels;
    // interpreting only the name can bind a null input or silently change math.
    if (low_precision_attention_operation(c.operation,p) != LowPrecisionAttention::None) return false;
    for (size_t i = 0; i < c.inputs.size(); ++i) if (c.inputs[i]) {
        if (c.inputs[i]->dtype != DType::Float32 || c.inputs[i]->backend != c.output->backend ||
            c.inputs[i]->device_ordinal() != c.output->device_ordinal()) return false;
        if (p.input_rank(KernelInput(i)) < 0 || p.input_rank(KernelInput(i)) > 8) return false;
        size_t count = 1;
        for (int d = p.input_rank(KernelInput(i)) - 1; d >= 0; --d) {
            if (p.input_shape(KernelInput(i), d) <= 0 || p.input_stride(KernelInput(i), d) != int(count)) return false;
            count *= size_t(p.input_shape(KernelInput(i), d));
            if (count > size_t(0x7fffffff)) return false;
        }
        if (count != c.inputs[i]->count) return false;
    }
    auto input = [&](int i) { return g.input(c.inputs.at(i), input_shape(p, i)); };
    auto add = [&](int a, int b) { return g.binary(Op::Add, a, b); };
    auto sub = [&](int a, int b) { return g.binary(Op::Subtract, a, b); };
    auto mul = [&](int a, int b) { return g.binary(Op::Multiply, a, b); };
    auto div = [&](int a, int b) { return g.binary(Op::Divide, a, b); };
    auto constant = [&](float f) { return g.constant(f); };
    int result = -1;
    static const std::map<std::string, Op> binary{{"add",Op::Add},{"sub",Op::Subtract},{"mul",Op::Multiply},{"div",Op::Divide},{"pow",Op::Power}};
    static const std::map<std::string, Op> unary{{"exp",Op::Exp},{"sqrt",Op::Sqrt},{"rsqrt",Op::Rsqrt},{"neg",Op::Negate}};
    if (auto found = binary.find(c.operation); found != binary.end()) {
        const bool wrapped = p.attribute(0) == 1 && p.input_rank(KernelInput::B) == -1 && c.operation != "div";
        if ((!wrapped && p.attribute(0)) || (!wrapped && (c.inputs.size() < 2 || !c.inputs[1]))) return false;
        result = g.binary(found->second, input(0), wrapped ? g.parameter(uint32_t(p.attribute(1))) : input(1));
    } else if (auto found = unary.find(c.operation); found != unary.end()) {
        result = g.unary(found->second, input(0));
    } else if (c.operation == "sum" || c.operation == "mean" || c.operation == "max") {
        if (!last_axis(p)) return false;
        const int width=p.input_shape(KernelInput::A,p.input_rank(KernelInput::A)-1);
        // Mean's short/vectorized paths use distinct ordered trees. Keep those
        // atomic until their schedules can be represented without reassociation.
        ReductionOrder order=c.operation=="sum" && width>=256 ? ReductionOrder::Block128 : ReductionOrder::Sequential;
        const bool cuda=c.output->backend==Backend::CUDA;
        if(cuda && c.operation=="mean" && width<=128) return false;
        if(cuda && c.operation=="mean" && width<=8192 && width%4==0) {
            auto power=[](int n) { int r=1; while(r<=n/2 && r<512) r*=2; return r; };
            const int columns=power(width/4), rows=power(p.output_count());
            int lanes=std::min(columns,32), height=std::min(rows,512/lanes);
            lanes=std::min(columns,512/height);
            if(lanes!=32 || (width+lanes-1)/lanes>=std::min(height*16,256)) return false;
            order=ReductionOrder::Vector4Warp32;
        }
        result = g.reduce(c.operation == "sum" ? Op::Sum : c.operation == "mean" ? Op::Mean : Op::Maximum, input(0),order);
    } else if (c.operation == "gelu") {
        const int x = input(0);
        if (p.attribute(1)) {
            result = mul(mul(constant(.5f), x), add(constant(1), g.unary(Op::Erf, mul(x, constant(.7071067811865475f)))));
        } else {
            const int cube = mul(mul(mul(constant(.044715f), x), x), x);
            const int t = g.unary(Op::Tanh, mul(constant(.7978845608028654f), add(x, cube)));
            result = mul(mul(constant(.5f),x), add(constant(1),t));
        }
    } else if (c.operation == "softmax" || c.operation == "attention_softmax") {
        if (!last_axis(p)) return false;
        int x = input(0);
        {
            if (c.operation == "attention_softmax") {
                if (p.attribute(2)) return false;
                x = add(mul(x, g.scalar(1.f / from_bits(p.attribute(1)))), input(1));
            } else if (p.attribute(1) != 0 && p.attribute(1) != 1) return false;
            const int maximum = g.reduce(Op::Maximum, x);
            const int exponential = g.unary(Op::Exp, sub(x, maximum));
            result = div(exponential, g.reduce(Op::Sum, exponential));
        }
    } else if (c.operation == "dropout") {
        const double probability = float64_random_parameter(p, 0);
        if (!(probability >= 0 && probability <= 1)) return false;
        result = input(0);
        if (p.attribute(3) && probability != 0) {
            float threshold = float(probability);
            if (double(threshold) < probability) threshold = std::nextafter(threshold, 1.f);
            if (probability == 1) threshold = std::numeric_limits<float>::infinity();
            const float scale = probability == 1 ? 0.f : p.attribute(9) ? float(1. / (1. - probability)) : 1.f / float(1. - probability);
            const int random = g.uniform(g.values[result].shape, uint32_t(p.attribute(0)), uint32_t(p.attribute(1)));
            const int keep = g.binary(Op::GreaterEqual, random, g.scalar(threshold));
            result = mul(result, g.ternary(Op::Select, keep, g.scalar(scale), constant(0)));
        }
    } else return false;
    if (result < 0 || elements(g.values[result].shape) != c.output->count) return false;
    Dimensions expected;
    for (int d = 0; d < p.output_rank(); ++d) expected.push_back(p.output_shape(d));
    if (elements(expected) != c.output->count) return false;
    if (g.values[result].shape != expected && c.operation != "sum" && c.operation != "mean" && c.operation != "max") return false;
    g.store(result,c.output);
    g.commands.push_back(c);
    return true;
}
} // namespace elman::fusion
