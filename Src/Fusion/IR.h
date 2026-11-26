#pragma once
#include "Core/Backend.h"
#include <string>
#include <vector>

namespace elman::fusion {
using Dimensions = std::vector<int>;
enum class Op {
    Input, Constant, Parameter, Add, Subtract, Multiply, Divide, Power, Fma,
    Exp, Tanh, Erf, Sqrt, Rsqrt, Negate, GreaterEqual, Select, Uniform,
    Sum, Dot, Maximum, Mean, MomentsMean, MomentsVariance
};
enum class ReductionOrder { Warp32, Sequential, Block128, Vector4Warp32 };
struct Value {
    Op op;
    Dimensions shape;
    std::vector<int> inputs;
    uint32_t bits = 0;
    int slot = -1;
};
struct Store { int value, binding; };
struct Command {
    std::string operation;
    BufferPtr output;
    std::vector<BufferPtr> inputs;
    KernelParams parameters;
};
struct Graph {
    std::vector<Value> values;
    std::vector<BufferPtr> bindings;
    std::vector<uint32_t> parameters;
    std::vector<Store> stores;
    std::vector<Command> commands;
    int input(const BufferPtr& buffer, const Dimensions& shape);
    int constant(float value);
    int parameter(uint32_t bits);
    int scalar(float value);
    int unary(Op op, int x);
    int binary(Op op, int x, int y);
    int ternary(Op op, int x, int y, int z);
    int reduce(Op op, int x, ReductionOrder order = ReductionOrder::Warp32);
    int dot(int x, int y);
    int uniform(const Dimensions& shape, uint32_t seed, uint32_t offset);
    void store(int value, const BufferPtr& buffer);
};
struct Schedule {
    Dimensions domain;
    bool reduction = false;
    bool vectorized_lanes = false;
    int rows = 0, width = 0, items = 1;
};
// Semantic lowering is separate from region selection and CUDA scheduling.
bool lower(Graph& graph, const Command& command);
bool has_definition(const std::string& operation);
bool analyze(const Graph& graph, Schedule& schedule);
std::string generate(const Graph& graph, const Schedule& schedule);
size_t elements(const Dimensions& shape);
bool broadcasts_to(const Dimensions& source, const Dimensions& target);
bool is_reduction(Op op);
} // namespace elman::fusion
