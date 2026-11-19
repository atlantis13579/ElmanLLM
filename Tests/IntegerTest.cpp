#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"

namespace {

namespace integer_extrema {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid integer extrema descriptor was accepted");
}
KernelParams params() {
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = 2;
    p.v[KernelParams::kOutputRankSlot] = 1;
    p.v[KernelParams::kOutputShapeSlot] = 2;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kInputRankSlot] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 1] = 1;
    p.v[KernelParams::kInputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kAttributeSlot] = 1;
    return p;
}
void malformed(Backend backend) {
    auto input = Tensor::from_int64_data({2, 3}, {INT64_MIN, -9007199254740993LL, 7,
                                               9007199254740993LL, INT64_MAX, -1}, backend).node()->value;
    for (const auto& op : {"max", "min", "amax", "amin"}) {
        const bool maximum = std::string(op) == "max" || std::string(op) == "amax";
        auto output = backend_allocate(backend, 2, DType::Int64);
        auto valid = params();
        backend_dispatch(op, output, {input}, valid);
        check(backend_download_int64(output) == (maximum ? std::vector<int64_t>{7, INT64_MAX} :
              std::vector<int64_t>{INT64_MIN, -1}), "Integer extremum lost exact ordering");
        const auto before = backend_download_raw(output);
        for (auto bad : std::vector<std::pair<int, int>>{
                {KernelParams::kInputRankSlot, -1}, {KernelParams::kInputRankSlot, 9},
                {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, -1},
                {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, 1},
                {KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset, 1},
                {KernelParams::kAttributeSlot, -2}, {KernelParams::kAttributeSlot, 2},
                {KernelParams::kAttributeSlot + 1, 2}, {KernelParams::kOutputCountSlot, 1},
                {KernelParams::kOutputRankSlot, 9}, {KernelParams::kOutputShapeSlot, 3},
                {KernelParams::kInputDTypeSlot, int(DType::Int32)}}) {
            auto p = valid; p.v[bad.first] = bad.second;
            rejects([&] { backend_dispatch(op, output, {input}, p); });
            check(backend_download_raw(output) == before, "Malformed extrema mutated output");
        }
        rejects([&] { backend_dispatch(op, output, {}, valid); });
        rejects([&] { backend_dispatch(op, output, {BufferPtr{}}, valid); });
        rejects([&] { backend_dispatch(op, output, {input, input}, valid); });
        auto narrow = backend_allocate(backend, 6, DType::Int32);
        auto p = valid; p.v[KernelParams::kInputDTypeSlot] = int(DType::Int32);
        rejects([&] { backend_dispatch(op, output, {narrow}, p); });
        auto empty = backend_allocate(backend, 0, DType::Int64);
        p = valid;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = 0;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 0;
        rejects([&] { backend_dispatch(op, output, {empty}, p); });
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = 0;
        p.v[KernelParams::kOutputShapeSlot] = p.v[KernelParams::kOutputCountSlot] = 0;
        rejects([&] { backend_dispatch(op, empty, {empty}, p); });
        p.v[KernelParams::kInputRankSlot] = 3;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = INT32_MAX;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = INT32_MAX;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 2] = 0;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 1] = 0;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 2] = 1;
        rejects([&] { backend_dispatch(op, empty, {empty}, p); });
    }
}
void values_and_aliases(Backend backend) {
    std::vector<uint64_t> values = {0, uint64_t(1) << 63, UINT64_MAX, 9007199254740993ULL};
    std::vector<uint8_t> bytes(values.size() * sizeof(uint64_t));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    auto unsigned_values = Tensor::from_raw({2, 2}, DType::UInt64, bytes, backend);
    auto high = dispatch_tensor("max", {}, {unsigned_values}, {-1, 0}, DType::UInt64).raw_data();
    check(high == std::vector<uint8_t>(8, 255), "Unsigned global maximum used signed ordering");
    auto low = dispatch_tensor("min", {1, 1}, {unsigned_values}, {-1, 1}, DType::UInt64).raw_data();
    check(low == std::vector<uint8_t>(8, 0), "Unsigned global minimum used signed ordering");
    auto boolean = Tensor::from_raw({2, 3}, DType::Bool, {0, 2, 128, 255, 1, 0}, backend);
    check(max(boolean, 1).raw_data() == std::vector<uint8_t>{1, 1}, "Bool maximum was not normalized");
    check(min(boolean, 1).raw_data() == std::vector<uint8_t>{0, 0}, "Bool minimum differs");
    auto root = Tensor::from_int64_data({10}, {INT64_MIN, 1, INT64_MAX, -9007199254740993LL, 2, 5, 0, 0, 0, 0}, backend);
    auto source = contiguous_view(root, {2, 3});
    auto overlap = contiguous_view(root, {2}, 1);
    auto disjoint = contiguous_view(root, {2}, 6);
    const auto before = root.raw_data();
    for (const auto& op : {"max", "min", "amax", "amin"}) {
        rejects([&] { dispatch_tensor_into(op, overlap, {source}, {1, 0}); });
        check(root.raw_data() == before, "Rejected overlapping extrema changed storage");
    }
    dispatch_tensor_into("amax", disjoint, {source}, {1, 0});
    check(disjoint.int64_data() == std::vector<int64_t>{INT64_MAX, 5}, "Disjoint extrema views differ");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 4; ++iteration) {
            Tensor value;
            {
                StreamScope scope(producer, asynchronous, true);
                auto input = Tensor::from_int64_data({2, 3}, {INT64_MIN, -9007199254740993LL, -9007199254740995LL,
                    9007199254740993LL, INT64_MAX, 0}, backend);
                value = max(input, 1);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                check(min(value, 0).int64_data() == std::vector<int64_t>{-9007199254740993LL}, "Cross-stream extrema lost exact values");
                check(amax(value, 0).int64_data() == std::vector<int64_t>{INT64_MAX}, "Cross-stream maximum lost its producer");
            }
        }
        producer.synchronize(); consumer.synchronize();
    }
}

void run(Backend backend) {
    const auto before = memory_stats(backend).live_bytes();
        malformed(backend); values_and_aliases(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Integer extrema leaked storage");
        std::cout << "Integer extrema: " << checks << " passed, 0 failed\n";
}

}  // namespace integer_extrema

namespace integer_matrix {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid integer matrix operation was accepted");
}
Tensor left(Backend backend) {
    return Tensor::from_int64_data({2, 3}, {INT64_MAX, 1, INT64_MIN, -1, 2, 3}, backend);
}
Tensor right(Backend backend) {
    return Tensor::from_int64_data({3, 2}, {1, 0, 0, 1, 1, 1}, backend);
}
const std::vector<int64_t> expected{-1, INT64_MIN + 1, 2, 5};

void malformed(Backend backend) {
    auto a = left(backend), b = right(backend);
    auto out = Tensor::from_int64_data({2, 2}, std::vector<int64_t>(4), backend);
    KernelParams p;
    for (auto input : {KernelInput::A, KernelInput::B, KernelInput::C, KernelInput::D})
        p.v[KernelParams::input_slot(input)] = -1;
    p.v[KernelParams::kOutputCountSlot] = 4;
    p.v[KernelParams::kOutputRankSlot] = 2;
    p.v[KernelParams::kOutputShapeSlot] = 2;
    p.v[KernelParams::kOutputShapeSlot + 1] = 2;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int64);
    for (auto input : {KernelInput::A, KernelInput::B}) {
        const int slot = KernelParams::input_slot(input);
        const bool first = input == KernelInput::A;
        p.v[slot] = 2;
        p.v[slot + KernelParams::kInputShapeOffset] = first ? 2 : 3;
        p.v[slot + KernelParams::kInputShapeOffset + 1] = first ? 3 : 2;
        p.v[slot + KernelParams::kInputStrideOffset] = first ? 3 : 2;
        p.v[slot + KernelParams::kInputStrideOffset + 1] = 1;
        p.v[KernelParams::input_dtype_slot(input)] = int(DType::Int64);
    }
    backend_dispatch("matmul", out.node()->value, {a.node()->value, b.node()->value}, p);
    check(out.int64_data() == expected, "Integer matrix overflow or low bits differ");
    const auto before = out.raw_data();
    for (auto mutation : std::vector<std::pair<int, int>>{
            {KernelParams::kOutputCountSlot, 3}, {KernelParams::kOutputRankSlot, -1}, {KernelParams::kOutputRankSlot, 9},
            {KernelParams::kOutputShapeSlot, -1}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kInputRankSlot, 0}, {KernelParams::kInputRankSlot, 9},
            {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, -1},
            {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, 1},
            {KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kAttributeSlot, -1}, {KernelParams::kAttributeSlot, 2},
            {KernelParams::kInputDTypeSlot, int(DType::Int32)}}) {
        auto bad = p; bad.v[mutation.first] = mutation.second;
        rejects([&] { backend_dispatch("matmul", out.node()->value, {a.node()->value, b.node()->value}, bad); });
        check(out.raw_data() == before, "Rejected integer matrix changed output");
    }
    rejects([&] { backend_dispatch("matmul", out.node()->value, {}, p); });
    rejects([&] { backend_dispatch("matmul", out.node()->value, {a.node()->value}, p); });
    rejects([&] { backend_dispatch("matmul", out.node()->value, {a.node()->value, b.node()->value, b.node()->value}, p); });
    rejects([&] { dispatch_tensor_into("matmul", a, {a, Tensor::from_int64_data({3, 3}, std::vector<int64_t>(9), backend)}); });
    auto backing = Tensor::from_int64_data({10}, {INT64_MAX, 1, INT64_MIN, -1, 2, 3, 0, 0, 0, 0}, backend);
    auto source = contiguous_view(backing, {2, 3}), target = contiguous_view(backing, {2, 2}, 6);
    dispatch_tensor_into("matmul", target, {source, b});
    check(target.int64_data() == expected, "Disjoint integer matrix aliases differ");
    check(addmm(out, a, b, .5f, 1.5f).int64_data() == expected, "Integer addmm scalar truncation differs");
    for (float coefficient : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN(), 9223372036854775808.0f}) {
        rejects([&] { addmm(out, a, b, 1, coefficient); });
        rejects([&] { addmm(out, a, b, coefficient, 1); });
    }
    auto byte = Tensor::from_raw({1, 1}, DType::UInt8, {2}, backend);
    rejects([&] { addmm(byte, byte, byte, 1, -1); });
    rejects([&] { addmm(byte, byte, byte, 1, 255.5f); });
    check(addmm(byte, byte, byte, 1, 255).raw_data() == std::vector<uint8_t>{254}, "Unsigned addmm did not wrap in its dtype");
    check(matmul(Tensor::from_int64_data({2, 0}, {}, backend), Tensor::from_int64_data({0, 3}, {}, backend)).int64_data() ==
          std::vector<int64_t>(6), "Zero-width integer dot product differs");
    rejects([&] { packed_linear(a, transpose(b, 0, 1), {1, 2}); });
    rejects([&] { linear(a, b); });
}
void execution(Backend backend) {
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 16; ++iteration) {
            Tensor product, packed;
            {
                StreamScope scope(producer, asynchronous, true);
                product = matmul(left(backend), right(backend));
                packed = packed_linear(left(backend), transpose(right(backend), 0, 1), {1, 0, 1});
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                auto copied = add(product, Tensor::from_int64_data({}, {0}, backend));
                product = {};
                check(copied.int64_data() == expected, "Integer matrix lost a released cross-stream producer");
                check(packed.int64_data() == std::vector<int64_t>{-1, 2, INT64_MIN + 1, 5}, "Integer packed projection layout differs");
            }
        }
        producer.synchronize(); consumer.synchronize();
    }
}
void empty_boolean(Backend backend) {
    AllocationPool pool(backend);
    ExecutionStream producer(backend), consumer(backend);
    auto a = Tensor::from_raw({2, 0}, DType::Bool, {}, backend);
    auto b = Tensor::from_raw({0, 4}, DType::Bool, {}, backend);
    auto bias = Tensor::from_raw({4}, DType::Bool, {0, 2, 128, 255}, backend);
    rejects([&] { matmul(Tensor::from_raw({0}, DType::Bool, {}, backend), Tensor::from_raw({0}, DType::Bool, {}, backend)); });
    rejects([&] { matmul(a, Tensor::from_raw({1, 4}, DType::Bool, {0, 0, 0, 0}, backend)); });
    if (backend == Backend::CUDA) {
        rejects([&] { matmul(a, Tensor::from_raw({0, 4}, DType::Bool, {}, Backend::CPU)); });
        rejects([&] { addmm(Tensor::from_raw({4}, DType::Bool, {0, 1, 1, 1}, Backend::CPU), a, b); });
    }
    for (int iteration = 0; iteration < 8; ++iteration) {
        Tensor product, projection, packed;
        {
            StreamScope scope(producer, true, true);
            product = matmul(a, b);
            projection = linear(a, transpose(b, 0, 1));
            packed = packed_linear(a, transpose(b, 0, 1), {1, 0, 3});
        }
        {
            StreamScope scope(consumer, true, true);
            check(logical_or(product, projection).raw_data() == std::vector<uint8_t>(8), "Empty Bool product lost stream ownership");
            check(packed.raw_data() == std::vector<uint8_t>(8), "Empty Bool packed product differs");
        }
    }
    if (backend == Backend::CUDA) {
        check(addmm(bias, a, b).raw_data() == std::vector<uint8_t>{0, 1, 1, 1, 0, 1, 1, 1}, "Bool addmm did not normalize its addend");
    } else {
        rejects([&] { addmm(bias, a, b); });
    }
    check(addmm(bias, a, b, 0).raw_data() == std::vector<uint8_t>(8), "Bool zero-beta addmm differs");
    producer.synchronize(); consumer.synchronize();
}

void run(Backend backend) {
    const auto before = memory_stats(backend).live_bytes();
        malformed(backend); execution(backend); empty_boolean(backend);
        check(memory_stats(backend).live_bytes() == before, "Integer matrix leaked allocation ownership");
        std::cout << "Integer matrix: " << checks << " passed, 0 failed\n";
}

}  // namespace integer_matrix

namespace integer_pointwise {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid integer pointwise operation was accepted");
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void malformed(Backend backend) {
    auto a = Tensor::from_int64_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    auto b = Tensor::from_int64_data({1, 3}, {1, 2, 3}, backend);
    auto out = Tensor::from_int64_data({2, 3}, std::vector<int64_t>(6), backend);
    const auto valid = descriptor(out, {a, b});
    const std::vector<BufferPtr> inputs = {a.node()->value, b.node()->value};
    backend_dispatch("add", out.node()->value, inputs, valid);
    check(out.int64_data() == std::vector<int64_t>{2, 4, 6, 5, 7, 9}, "Raw integer broadcast differs");
    const auto before = out.raw_data();
    const int a_slot = KernelParams::input_slot(KernelInput::A), b_slot = KernelParams::input_slot(KernelInput::B);
    for (auto bad : std::vector<std::pair<int, int>>{
            {a_slot, -1}, {a_slot, 9}, {b_slot, -1}, {b_slot, 9},
            {a_slot + KernelParams::kInputShapeOffset, -1}, {a_slot + KernelParams::kInputShapeOffset, 3},
            {b_slot + KernelParams::kInputShapeOffset, 2},
            {a_slot + KernelParams::kInputStrideOffset, 1}, {b_slot + KernelParams::kInputStrideOffset + 1, 0},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, -1},
            {KernelParams::kOutputRankSlot, 9}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kOutputDTypeSlot, int(DType::Int32)}, {KernelParams::kInputDTypeSlot, int(DType::Int32)}}) {
        auto p = valid; p.v[bad.first] = bad.second;
        rejects([&] { backend_dispatch("add", out.node()->value, inputs, p); });
        check(out.raw_data() == before, "Rejected descriptor mutated output");
    }
    rejects([&] { backend_dispatch("add", out.node()->value, {}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0]}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], {}}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], inputs[1], inputs[1]}, valid); });
    auto narrow = cast(b, DType::Int32);
    auto mixed = descriptor(out, {a, narrow});
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], narrow.node()->value}, mixed); });
    auto floating = cast(out, DType::Float32);
    auto wrong = descriptor(floating, {a, b});
    rejects([&] { backend_dispatch("add", floating.node()->value, inputs, wrong); });
    auto empty = Tensor::from_int64_data({0}, {}, backend);
    auto p = descriptor(empty, {empty});
    p.v[a_slot] = 3;
    p.v[a_slot + KernelParams::kInputShapeOffset] = INT32_MAX;
    p.v[a_slot + KernelParams::kInputShapeOffset + 1] = INT32_MAX;
    p.v[a_slot + KernelParams::kInputShapeOffset + 2] = 0;
    p.v[a_slot + KernelParams::kInputStrideOffset] = 0;
    p.v[a_slot + KernelParams::kInputStrideOffset + 1] = 0;
    p.v[a_slot + KernelParams::kInputStrideOffset + 2] = 1;
    rejects([&] { backend_dispatch("abs", empty.node()->value, {empty.node()->value}, p); });
}
void aliases(Backend backend) {
    auto root = Tensor::from_int64_data({12}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, backend);
    auto source = contiguous_view(root, {2, 3});
    auto ones = Tensor::from_int64_data({2, 3}, std::vector<int64_t>(6, 1), backend);
    dispatch_tensor_into("add", source, {source, ones});
    check(source.int64_data() == std::vector<int64_t>{2, 3, 4, 5, 6, 7}, "Exact in-place integer operation differs");
    auto shifted = contiguous_view(root, {2, 3}, 1);
    auto before = root.raw_data();
    rejects([&] { dispatch_tensor_into("add", shifted, {source, ones}); });
    rejects([&] { dispatch_tensor_into("add", source, {contiguous_view(root, {1, 3}), ones}); });
    check(root.raw_data() == before, "Rejected overlap mutated storage");
    auto disjoint = contiguous_view(root, {2, 3}, 6);
    dispatch_tensor_into("mul", disjoint, {source, source});
    check(disjoint.int64_data() == std::vector<int64_t>{4, 9, 16, 25, 36, 49}, "Disjoint aliases differ");
    auto boolean = Tensor::from_raw({5}, DType::Bool, {0, 1, 2, 128, 255}, backend);
    check(cast(boolean, DType::Int64).int64_data() == std::vector<int64_t>{0, 1, 1, 1, 1}, "Noncanonical Bool conversion differs");
    check(logical_not(boolean).raw_data() == std::vector<uint8_t>{1, 0, 0, 0, 0}, "Noncanonical Bool predicate differs");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (int iteration = 0; iteration < 4; ++iteration) {
            Tensor value;
            {
                StreamScope scope(producer, asynchronous, true);
                auto input = Tensor::from_int64_data({6}, {9007199254740993LL, -9007199254740993LL,
                    32767, -32768, INT64_MAX, INT64_MIN}, backend);
                value = mul(input, Tensor::from_int64_data({}, {1}, backend));
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                auto incremented = add(value, Tensor::from_int64_data({}, {1}, backend));
                auto result = cast(cast(incremented, DType::Int16), DType::Int64);
                check(result.int64_data() == std::vector<int64_t>{2, 0, -32768, -32767, 0, 1},
                      "Integer pointwise stream dependencies or narrowing differ");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Integer pointwise pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const auto before = memory_stats(backend).live_bytes();
        malformed(backend); aliases(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Integer pointwise operations leaked storage");
        std::cout << "Integer pointwise: " << checks << " passed, 0 failed\n";
}

}  // namespace integer_pointwise

namespace integer_reduction {

using namespace elman;
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid integer operation was accepted");
}
void typed_access(Backend backend) {
    const std::vector<int64_t> exact = {INT64_MIN, INT64_MAX, 9007199254740993LL, -9007199254740993LL, -1, 0, 1};
    auto tensor = Tensor::from_int64_data({7}, exact, backend);
    check(tensor.dtype() == DType::Int64 && !tensor.requires_grad(), "Int64 constructor metadata differs");
    check(tensor.int64_data() == exact, "Int64 constructor rounded values");
    check(tensor.to(Backend::CPU).int64_data() == exact, "Int64 transfer rounded values");
    auto view = contiguous_view(tensor, {3}, 2);
    auto version = tensor.node()->value->version;
    view.assign_int64({17, INT64_MIN, INT64_MAX});
    auto changed = exact;
    changed[2] = 17; changed[3] = INT64_MIN; changed[4] = INT64_MAX;
    check(tensor.int64_data() == changed, "Typed view assignment escaped its range");
    check(tensor.node()->value->version == version + 1, "Typed assignment lost its shared version");
    rejects([&] { view.assign_int64({1, 2}); });
    check(tensor.node()->value->version == version + 1 && tensor.int64_data() == changed,
          "Rejected assignment changed bytes or version");
    auto narrow = Tensor::from_int_data({1}, {1}, backend);
    rejects([&] { narrow.assign_int64({1}); });
    rejects([&] { narrow.int64_data(); });
    rejects([&] { Tensor::from_int64_data({2}, {1}, backend); });
    rejects([&] { Tensor().int64_data(); });
    rejects([&] { backend_download_int64({}); });
    check(Tensor::from_int64_data({0}, {}, backend).int64_data().empty(), "Empty typed access differs");
}
void arithmetic(Backend backend) {
    auto x = Tensor::from_int64_data({2, 3}, {INT64_MAX, 1, 1, INT64_MIN, -1, 1}, backend);
    check(sum(x, 1).int64_data() == std::vector<int64_t>{INT64_MIN + 1, INT64_MIN}, "Sum overflow differs");
    check(prod(x, 1).int64_data() == std::vector<int64_t>{INT64_MAX, INT64_MIN}, "Product overflow differs");
    check(cumsum(x, 1).int64_data() == std::vector<int64_t>{INT64_MAX, INT64_MIN, INT64_MIN + 1,
                                                         INT64_MIN, INT64_MAX, INT64_MIN}, "Scan overflow differs");
    check(sum_all(x).int64_data() == std::vector<int64_t>{1}, "Global sum overflow differs");
    auto scalar = Tensor::from_int64_data({}, {9007199254740993LL}, backend);
    for (int axis : {-1, 0}) for (bool keep : {false, true}) {
        auto value = sum(scalar, axis, keep);
        check(value.shape().empty() && value.int64_data() == scalar.int64_data(), "Scalar sum changed shape/value");
        check(prod(scalar, axis, keep).int64_data() == scalar.int64_data(), "Scalar product changed value");
    }
    auto boolean = Tensor::from_raw({5}, DType::Bool, {0, 1, 2, 128, 255}, backend);
    check(sum_all(boolean).int64_data() == std::vector<int64_t>{4}, "Boolean reduction did not normalize nonzero bytes");
    check(cumsum(boolean).int64_data() == std::vector<int64_t>{0, 1, 2, 3, 4}, "Boolean scan did not normalize nonzero bytes");
    for (auto dtype : {DType::Bool, DType::Int8, DType::UInt8, DType::Int16, DType::UInt16,
                       DType::Int32, DType::UInt32, DType::Int64, DType::UInt64}) {
        auto empty = Tensor::from_raw({2, 0, 3}, dtype, {}, backend);
        auto total = sum(empty, 1), product = prod(empty, 1, true);
        check(total.shape() == Shape{2, 3} && total.int64_data() == std::vector<int64_t>(6, 0), "Empty sum identity differs");
        check(product.shape() == Shape{2, 1, 3} && product.int64_data() == std::vector<int64_t>(6, 1), "Empty product identity differs");
        check(cumsum(empty, 1).shape() == empty.shape() && cumsum(empty, 0).int64_data().empty(), "Empty scan differs");
        check(sum(empty, 0).int64_data().empty() && prod(empty, 2).int64_data().empty(), "Empty output differs");
        check(sum_all(empty).int64_data() == std::vector<int64_t>{0}, "Global empty sum differs");
    }
}
KernelParams reduction_params(bool scan = false) {
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = scan ? 6 : 2;
    p.v[KernelParams::kOutputRankSlot] = scan ? 2 : 1;
    p.v[KernelParams::kOutputShapeSlot] = 2;
    if (scan) p.v[KernelParams::kOutputShapeSlot + 1] = 3;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kInputRankSlot] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 1] = 1;
    p.v[KernelParams::kInputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kAttributeSlot] = 1;
    return p;
}
void malformed(Backend backend) {
    auto input = Tensor::from_int64_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend).node()->value;
    {
        auto narrow = Tensor::from_int_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend).node()->value;
        auto output = backend_allocate(backend, 2, DType::Int32);
        auto p = reduction_params();
        p.v[KernelParams::kInputDTypeSlot] = p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int32);
        backend_dispatch("sum", output, {narrow}, p);
        check(backend_download_int(output) == std::vector<int32_t>{6, 15}, "Explicit raw Int32 sum contract changed");
        backend_dispatch("prod", output, {narrow}, p);
        check(backend_download_int(output) == std::vector<int32_t>{6, 120}, "Explicit raw Int32 product contract changed");
    }
    for (auto op : {"sum", "prod", "cumsum"}) {
        bool scan = std::string(op) == "cumsum";
        auto output = backend_allocate(backend, scan ? 6 : 2, DType::Int64);
        auto valid = reduction_params(scan);
        backend_dispatch(op, output, {input}, valid);
        auto previous = backend_download_raw(output);
        for (auto bad : std::vector<std::pair<int, int>>{
                {KernelParams::kInputRankSlot, -1}, {KernelParams::kInputRankSlot, 9},
                {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, -1},
                {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, 1},
                {KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset, 1},
                {KernelParams::kAttributeSlot, -2}, {KernelParams::kAttributeSlot, 2},
                {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, 9},
                {KernelParams::kOutputShapeSlot, 3}, {KernelParams::kInputDTypeSlot, int(DType::Int32)}}) {
            auto p = valid; p.v[bad.first] = bad.second;
            rejects([&] { backend_dispatch(op, output, {input}, p); });
            check(backend_download_raw(output) == previous, "Malformed descriptor mutated output");
        }
        if (!scan) {
            auto p = valid; p.v[KernelParams::kAttributeSlot + 1] = 2;
            rejects([&] { backend_dispatch(op, output, {input}, p); });
        }
        rejects([&] { backend_dispatch(op, output, {}, valid); });
        rejects([&] { backend_dispatch(op, output, {BufferPtr{}}, valid); });
        rejects([&] { backend_dispatch(op, output, {input, input}, valid); });
        auto floating = backend_allocate(backend, 6, DType::Float32);
        auto p = valid; p.v[KernelParams::kInputDTypeSlot] = int(DType::Float32);
        rejects([&] { backend_dispatch(op, output, {floating}, p); });
    }
    auto root = Tensor::from_int64_data({10}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0}, backend);
    {
        auto empty = backend_allocate(backend, 0, DType::Int64);
        auto scalar = backend_allocate(backend, 1, DType::Int64);
        auto p = reduction_params();
        p.v[KernelParams::kInputRankSlot] = 3;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = INT32_MAX;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = INT32_MAX;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 2] = 0;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 0;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 1] = 0;
        p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 2] = 1;
        p.v[KernelParams::kOutputRankSlot] = 0;
        p.v[KernelParams::kOutputCountSlot] = 1;
        p.v[KernelParams::kAttributeSlot] = -1;
        rejects([&] { backend_dispatch("sum", scalar, {empty}, p); });
        rejects([&] { backend_dispatch("prod", scalar, {empty}, p); });
    }
    auto values = contiguous_view(root, {2, 3});
    auto overlap = contiguous_view(root, {2}, 1);
    rejects([&] { dispatch_tensor_into("sum", overlap, {values}, {1, 0}); });
    rejects([&] { dispatch_tensor_into("prod", overlap, {values}, {1, 0}); });
    rejects([&] { dispatch_tensor_into("cumsum", values, {values}, {1}); });
    auto disjoint = contiguous_view(root, {2}, 6);
    dispatch_tensor_into("sum", disjoint, {values}, {1, 0});
    check(disjoint.int64_data() == std::vector<int64_t>{6, 15}, "Disjoint reduction aliases were rejected/corrupted");
}
void execution(Backend backend) {
    for (bool pooled : {false, true}) for (bool async : {false, true}) {
        ExecutionStream producer(backend), consumer(backend);
        AllocationPool pool(backend, 0, pooled);
        Tensor scan;
        {
            StreamScope scope(producer, async, true);
            auto source = Tensor::from_int64_data({2, 3}, {9007199254740993LL, 1, -1, 7, 9, 11}, backend);
            scan = cumsum(source, 1);
        }
        {
            StreamScope scope(consumer, async, true);
            auto total = sum(scan, 1);
            auto product = prod(scan, 1);
            check(total.int64_data() == std::vector<int64_t>{27021597764222980LL, 50}, "Cross-stream sum lost the scan producer");
            check(product.int64_data()[1] == 3024, "Cross-stream product lost the scan producer");
        }
        producer.synchronize(); consumer.synchronize();
    }
}

void run(Backend backend) {
    auto before = memory_stats(backend).live_bytes();
        typed_access(backend); arithmetic(backend); malformed(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Integer reductions leaked storage");
        std::cout << "Integer reductions: " << checks << " passed, 0 failed\n";
}

}  // namespace integer_reduction

namespace integer_sort {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid sort descriptor was accepted");
}
void malformed(Backend backend) {
    auto a = Tensor::from_int64_data({2, 3}, {INT64_MAX, 0, INT64_MIN, -1, -1, 1}, backend);
    auto out = Tensor::from_int_data({2, 3}, std::vector<int32_t>(6), backend);
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = 6;
    p.v[KernelParams::kOutputRankSlot] = 2;
    p.v[KernelParams::kOutputShapeSlot] = 2;
    p.v[KernelParams::kOutputShapeSlot + 1] = 3;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int32);
    p.v[KernelParams::kInputRankSlot] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 1] = 1;
    p.v[KernelParams::kInputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kAttributeSlot] = 1;
    backend_dispatch("argsort", out.node()->value, {a.node()->value}, p);
    check(out.int_data() == std::vector<int32_t>{2, 1, 0, 0, 1, 2}, "Exact signed stable sort differs");
    const auto before = out.raw_data();
    for (auto mutation : std::vector<std::pair<int, int>>{
            {KernelParams::kOutputCountSlot, 5}, {KernelParams::kOutputRankSlot, -1}, {KernelParams::kOutputRankSlot, 9},
            {KernelParams::kOutputShapeSlot, -1}, {KernelParams::kOutputShapeSlot, 3},
            {KernelParams::kInputRankSlot, 0}, {KernelParams::kInputRankSlot, 9},
            {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, -1},
            {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, 1},
            {KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kAttributeSlot, -1}, {KernelParams::kAttributeSlot, 2},
            {KernelParams::kAttributeSlot + 1, 2}, {KernelParams::kInputDTypeSlot, int(DType::Int32)}}) {
        auto bad = p; bad.v[mutation.first] = mutation.second;
        rejects([&] { backend_dispatch("argsort", out.node()->value, {a.node()->value}, bad); });
        check(out.raw_data() == before, "Rejected sort changed output");
    }
    rejects([&] { backend_dispatch("argsort", out.node()->value, {}, p); });
    rejects([&] { backend_dispatch("argsort", out.node()->value, {a.node()->value, a.node()->value}, p); });
    rejects([&] { dispatch_tensor("argsort", {2, 3}, {a}, {1, 0}, DType::Int64); });
    rejects([&] { dispatch_tensor_into("argsort", out, {out}, {1, 0}); });
    auto root = Tensor::from_int_data({12}, {3, 1, 2, 6, 4, 5, 0, 0, 0, 0, 0, 0}, backend);
    auto source = contiguous_view(root, {2, 3}), target = contiguous_view(root, {2, 3}, 6);
    dispatch_tensor_into("argsort", target, {source}, {1, 0});
    check(target.int_data() == std::vector<int32_t>{1, 2, 0, 1, 2, 0}, "Disjoint sort aliases differ");
}
void execution(Backend backend) {
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        for (int length : {1, 17, 257, 1025}) for (bool descending : {false, true}) {
            const std::vector<uint64_t> pattern{UINT64_MAX, 0, uint64_t(1) << 63, 0, (uint64_t(1) << 63) + 1};
            std::vector<uint64_t> data(length);
            for (int i = 0; i < length; ++i) data[i] = pattern[i % pattern.size()];
            std::vector<int32_t> indices;
            std::vector<uint64_t> ordered;
            auto levels = std::vector<uint64_t>{0, uint64_t(1) << 63, (uint64_t(1) << 63) + 1, UINT64_MAX};
            if (descending) std::reverse(levels.begin(), levels.end());
            for (auto value : levels) for (int i = 0; i < length; ++i) if (data[i] == value) {
                indices.push_back(i); ordered.push_back(value);
            }
            std::vector<uint8_t> bytes(data.size() * 8), expected(ordered.size() * 8);
            std::memcpy(bytes.data(), data.data(), bytes.size());
            std::memcpy(expected.data(), ordered.data(), expected.size());
            std::pair<Tensor, Tensor> sorted;
            {
                StreamScope scope(producer, asynchronous, true);
                sorted = sort(Tensor::from_raw({length}, DType::UInt64, bytes, backend), 0, descending);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                check(sorted.second.int_data() == indices, "Stable unsigned sort lost producer indices");
                check(sorted.first.raw_data() == expected, "Exact unsigned sort lost values");
            }
        }
        producer.synchronize(); consumer.synchronize();
    }
}

void run(Backend backend) {
    const auto before = memory_stats(backend).live_bytes();
        malformed(backend); execution(backend);
        check(memory_stats(backend).live_bytes() == before, "Sort leaked allocation ownership");
        std::cout << "Integer sort: " << checks << " passed, 0 failed\n";
}

}  // namespace integer_sort

namespace packed_integer {

using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid packed integer operation was accepted");
}
int narrow(int64_t value, DType dtype) {
    const auto& info = dtype_info(dtype);
    const int modulus = 1 << info.bits;
    int result = int(value % modulus);
    if (result < 0) result += modulus;
    if (info.kind == DTypeKind::SignedInteger && result >= modulus / 2) result -= modulus;
    return result;
}
KernelParams descriptor(const Tensor& output, const std::vector<Tensor>& inputs) {
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = int(output.numel());
    p.v[KernelParams::kOutputRankSlot] = int(output.shape().size());
    p.v[KernelParams::kOutputDTypeSlot] = int(output.dtype());
    for (size_t d = 0; d < output.shape().size(); ++d) p.v[KernelParams::kOutputShapeSlot + d] = output.shape()[d];
    for (int i = 0; i < 4; ++i) p.v[KernelParams::input_slot(KernelInput(i))] = -1;
    for (size_t i = 0; i < inputs.size(); ++i) {
        const auto input = KernelInput(i);
        const int slot = KernelParams::input_slot(input);
        p.v[slot] = int(inputs[i].shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(inputs[i].dtype());
        int stride = 1;
        for (int d = int(inputs[i].shape().size()) - 1; d >= 0; --d) {
            p.v[slot + KernelParams::kInputShapeOffset + d] = inputs[i].shape()[d];
            p.v[slot + KernelParams::kInputStrideOffset + d] = stride;
            stride *= inputs[i].shape()[d];
        }
    }
    return p;
}
void transport(Backend backend, DType dtype) {
    for (int count : {0, 1, 2, 3, 7, 8, 9, 15, 31, 257, 2051}) {
        std::vector<int32_t> integers(count), expected(count);
        std::vector<float> values(count), floating(count);
        for (int i = 0; i < count; ++i) {
            integers[i] = i - 1025;
            values[i] = float(integers[i]) + .25f;
            expected[i] = narrow(integers[i], dtype);
            floating[i] = float(narrow(int(values[i]), dtype));
        }
        auto value = Tensor::from_int_data({count}, integers, backend, dtype);
        check(value.int_data() == expected, "Packed integer host round trip differs");
        check(value.nbytes() == (size_t(count) * dtype_info(dtype).bits + 7) / 8, "Packed integer allocation size differs");
        const auto bytes = value.raw_data();
        check(value.to(Backend::CPU).raw_data() == bytes, "Packed integer backend transport differs");
        if (count * dtype_info(dtype).bits % 8)
            check((bytes.back() >> (count * dtype_info(dtype).bits % 8)) == 0, "Packed numeric encoding retained tail garbage");
        auto f = Tensor::from_data({count}, values, backend, false, dtype);
        check(f.data() == floating, "Packed float host conversion differs");
        const auto version = value.node()->value->version;
        value.assign(values);
        check(value.data() == floating && value.node()->value->version == version + 1, "Packed float assignment differs");
        value.assign_int(integers);
        check(value.int_data() == expected && value.node()->value->version == version + 2, "Packed integer assignment differs");
        check(Tensor::zeros({count}, backend, false, dtype).int_data() == std::vector<int32_t>(count), "Packed zeros differ");
        check(Tensor::ones({count}, backend, false, dtype).int_data() == std::vector<int32_t>(count, narrow(1, dtype)), "Packed ones differ");
    }
    std::vector<int64_t> wide = {INT64_MIN, INT64_MAX, 9007199254740993LL, -9007199254740993LL, -1, 0, 1};
    std::vector<int32_t> expected;
    for (int64_t value : wide) expected.push_back(narrow(value, dtype));
    check(cast(Tensor::from_int64_data({int(wide.size())}, wide, backend), dtype).int_data() == expected,
          "Packed Int64 conversion lost exact low bits");
    auto padding = Tensor::from_raw({1}, dtype, {255}, backend);
    auto copy = dispatch_tensor("cast", {1}, {padding}, {}, dtype);
    check(copy.raw_data() == std::vector<uint8_t>{255}, "Same-type packed cast changed padding");
    check(reshape(padding, {1, 1}).raw_data() == padding.raw_data(), "Packed storage reshape changed padding");
    rejects([&] { Tensor::from_raw({1}, dtype, {0}, backend, true); });
}
void malformed(Backend backend, DType dtype) {
    auto a = Tensor::from_int_data({2, 3}, {-7, -3, -1, 0, 1, 3}, backend, dtype);
    auto b = Tensor::from_int_data({1, 3}, {0, 1, 2}, backend, dtype);
    auto out = Tensor::zeros({2, 3}, backend, false, dtype);
    const auto valid = descriptor(out, {a, b});
    const std::vector<BufferPtr> inputs{a.node()->value, b.node()->value};
    const auto before = out.raw_data();
    const int slot = KernelParams::input_slot(KernelInput::A);
    for (auto field : std::vector<std::pair<int, int>>{
            {slot, -1}, {slot, 9}, {slot + 1, -1}, {slot + 1, 3}, {slot + 9, 1},
            {KernelParams::kOutputCountSlot, 1}, {KernelParams::kOutputRankSlot, 9},
            {KernelParams::kOutputShapeSlot, 3}, {KernelParams::kOutputDTypeSlot, int(DType::Int32)},
            {KernelParams::input_dtype_slot(KernelInput::A), int(DType::Int32)}}) {
        auto p = valid; p.v[field.first] = field.second;
        rejects([&] { backend_dispatch("add", out.node()->value, inputs, p); });
        check(out.raw_data() == before, "Rejected packed descriptor mutated output");
    }
    rejects([&] { backend_dispatch("add", out.node()->value, {}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0]}, valid); });
    rejects([&] { backend_dispatch("add", out.node()->value, {inputs[0], inputs[1], inputs[0]}, valid); });
    rejects([&] { add(a, cast(b, DType::Int32)); });
    rejects([&] { mean(a, 0); });
    rejects([&] { dispatch_tensor("cast", a.shape(), {a}, {}, DType::Complex64); });
    rejects([&] { dispatch_tensor("cast", a.shape(), {a}, {}, DType::QInt8); });
    rejects([&] { gather(a, 1, Tensor::from_int_data({2, 1}, {-1, 3}, backend)); });
    auto bad_cast = descriptor(out, {a});
    bad_cast.v[slot + KernelParams::kInputStrideOffset] = 1;
    rejects([&] { backend_dispatch("cast", out.node()->value, {inputs[0]}, bad_cast); });
    auto ids = Tensor::from_int_data({2, 3}, {0, 1, 2, 0, 1, 2}, backend);
    auto p = descriptor(out, {a, ids, a});
    p.v[KernelParams::kAttributeSlot] = 2;
    rejects([&] { backend_dispatch("scatter_add", out.node()->value, {inputs[0], ids.node()->value, inputs[0]}, p); });
    check(out.raw_data() == before, "Rejected packed operation mutated output");
}
void execution(Backend backend, const std::vector<DType>& formats) {
    for (bool pooled : {false, true}) for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream producer(backend), consumer(backend);
        for (auto dtype : formats) for (int iteration = 0; iteration < 3; ++iteration) {
            Tensor value;
            std::vector<int32_t> data{63, -64, -1, 0, 1, 7, -9, 127, 128};
            {
                StreamScope scope(producer, asynchronous, true);
                auto input = Tensor::from_int_data({3, 3}, data, backend, dtype);
                value = mul(input, input);
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                auto result = add(value, Tensor::ones({}, backend, false, dtype));
                for (auto& x : data) { const int n = narrow(x, dtype); x = narrow(n * n + 1, dtype); }
                check(result.int_data() == data, "Packed stream dependency or temporary lifetime differs");
                auto ids = Tensor::from_int_data({3, 3}, {0, 1, 2, 0, 1, 2, 0, 1, 2}, backend);
                auto added = dispatch_tensor("scatter_add", result.shape(), {result, ids, result}, {1}, dtype);
                for (auto& x : data) x = narrow(x * 2, dtype);
                check(added.int_data() == data, "Packed scatter-add stream result differs");
            }
        }
        producer.synchronize(); consumer.synchronize();
        if (pooled) check(pool.stats().reuses > 0, "Packed integer pool reuse was not exercised");
    }
}

void run(Backend backend) {
    const auto before = memory_stats(backend).live_bytes();
        std::vector<DType> formats;
        for (auto dtype : all_dtypes()) if (is_packed_integer(dtype)) formats.push_back(dtype);
        check(formats.size() == 14, "Packed dtype inventory differs");
        for (auto dtype : formats) { transport(backend, dtype); malformed(backend, dtype); }
        execution(backend, formats);
        check(memory_stats(backend).live_bytes() == before, "Packed integer operations leaked storage");
        std::cout << "Packed integer module: " << checks << " passed, 0 failed\n";
}

}  // namespace packed_integer

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto backend = elman::parse_backend(argc > 1 ? argv[1] : "cpu");
        if (!elman::backend_available(backend)) return 77;
        integer_extrema::run(backend);
        integer_matrix::run(backend);
        integer_pointwise::run(backend);
        integer_reduction::run(backend);
        integer_sort::run(backend);
        packed_integer::run(backend);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
