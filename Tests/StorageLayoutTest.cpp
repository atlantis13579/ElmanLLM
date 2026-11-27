#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"
#include <cstring>
#include <iostream>
#include <stdexcept>

namespace {
using namespace elman;
int checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, "Invalid storage layout was accepted");
}
void malformed(Backend backend) {
    Tensor a = Tensor::from_int64_data({2, 3}, {1, 2, 3, 4, 5, 6}, backend);
    Tensor out = Tensor::from_int64_data({3, 2}, std::vector<int64_t>(6), backend);
    KernelParams p;
    p.v[KernelParams::kOutputCountSlot] = 6;
    p.v[KernelParams::kOutputRankSlot] = 2;
    p.v[KernelParams::kOutputShapeSlot] = 3;
    p.v[KernelParams::kOutputShapeSlot + 1] = 2;
    p.v[KernelParams::kOutputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kInputRankSlot] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset] = 2;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset + 1] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset] = 3;
    p.v[KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset + 1] = 1;
    p.v[KernelParams::kInputDTypeSlot] = int(DType::Int64);
    p.v[KernelParams::kAttributeSlot] = 1;
    p.v[KernelParams::kAttributeSlot + 1] = 0;
    // Mark the unused operand descriptors absent. Metal validates every slot,
    // while the zero-initialized KernelParams default leaves ranks at 0.
    for (int slot = 1; slot < 4; ++slot)
        p.v[KernelParams::input_slot(static_cast<KernelInput>(slot))] = -1;
    backend_dispatch("permute", out.node()->value, {a.node()->value}, p);
    check(out.int64_data() == std::vector<int64_t>{1, 4, 2, 5, 3, 6}, "Storage permutation differs");
    const std::vector<uint8_t> before = out.raw_data();
    for (std::pair<int, int> mutation : std::vector<std::pair<int, int>>{
            {KernelParams::kOutputCountSlot, 5}, {KernelParams::kOutputRankSlot, -1}, {KernelParams::kOutputRankSlot, 9},
            {KernelParams::kOutputShapeSlot, -1}, {KernelParams::kOutputShapeSlot, 2},
            {KernelParams::kInputRankSlot, -1}, {KernelParams::kInputRankSlot, 9},
            {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, -1},
            {KernelParams::kInputRankSlot + KernelParams::kInputShapeOffset, 1},
            {KernelParams::kInputRankSlot + KernelParams::kInputStrideOffset, 1},
            {KernelParams::kAttributeSlot, -1}, {KernelParams::kAttributeSlot, 2},
            {KernelParams::kAttributeSlot + 1, 1}, {KernelParams::kInputDTypeSlot, int(DType::Int32)}}) {
        KernelParams bad = p; bad.v[mutation.first] = mutation.second;
        rejects([&] { backend_dispatch("permute", out.node()->value, {a.node()->value}, bad); });
        check(out.raw_data() == before, "Rejected layout changed output bytes");
    }
    rejects([&] { backend_dispatch("permute", out.node()->value, {}, p); });
    rejects([&] { backend_dispatch("permute", out.node()->value, {a.node()->value, a.node()->value}, p); });
    rejects([&] { dispatch_tensor_into("permute", reshape(a, {3, 2}), {a}, {1, 0}); });
    Tensor root = Tensor::from_int64_data({12}, {1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 0, 0}, backend);
    Tensor source = contiguous_view(root, {2, 3}), target = contiguous_view(root, {3, 2}, 6);
    dispatch_tensor_into("permute", target, {source}, {1, 0});
    check(target.int64_data() == std::vector<int64_t>{1, 4, 2, 5, 3, 6}, "Disjoint storage aliases differ");
    Tensor ids = Tensor::from_int_data({2}, {-1, 0}, backend);
    rejects([&] { dispatch_tensor("index_select", {2, 2}, {a, ids}, {1}, DType::Int64); });
    Tensor floating = Tensor::from_raw({2, 2}, DType::Float8E8M0FNU, {0, 1, 2, 3}, backend);
    check(triu(floating).raw_data() == std::vector<uint8_t>({0, 1, 0, 3}), "E8M0 triangular zero encoding differs");
    Tensor quantized = Tensor::from_raw({2, 2}, DType::QUInt8, {0, 1, 2, 3}, backend);
    rejects([&] { tril(quantized); });
}
template<class T> void integer_constructor(Backend backend, DType dtype) {
    const std::vector<int32_t> data = {INT32_MIN, -65537, -129, -1, 0, 1, 127, 256, 65537, INT32_MAX};
    std::vector<T> converted(data.begin(), data.end());
    std::vector<uint8_t> expected(converted.size() * sizeof(T));
    std::memcpy(expected.data(), converted.data(), expected.size());
    const Tensor value = Tensor::from_int_data({int(data.size())}, data, backend, dtype);
    check(value.raw_data() == expected, "Integer constructor changed exact storage bits");
    check(Tensor::from_int_data({0}, {}, backend, dtype).raw_data().empty(), "Empty integer constructor differs");
}
void integer_constructors(Backend backend) {
    integer_constructor<int8_t>(backend, DType::Int8);
    integer_constructor<uint8_t>(backend, DType::UInt8);
    integer_constructor<int16_t>(backend, DType::Int16);
    integer_constructor<uint16_t>(backend, DType::UInt16);
    integer_constructor<int32_t>(backend, DType::Int32);
    integer_constructor<uint32_t>(backend, DType::UInt32);
    integer_constructor<int64_t>(backend, DType::Int64);
    integer_constructor<uint64_t>(backend, DType::UInt64);
}
void execution(Backend backend) {
    for (bool asynchronous : {false, true}) {
        AllocationPool pool(backend);
        ExecutionStream producer(backend), consumer(backend);
        for (elman::DType dtype : all_dtypes()) {
            std::vector<uint8_t> bytes(dtype_storage_bytes(dtype, 30));
            for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = uint8_t((i * 37) ^ 0xa5);
            const unsigned remaining = unsigned(30 * dtype_info(dtype).bits % 8);
            if (remaining) bytes.back() &= uint8_t((1u << remaining) - 1);
            Tensor value;
            {
                StreamScope scope(producer, asynchronous, true);
                value = permute(Tensor::from_raw({2, 3, 5}, dtype, bytes, backend), {2, 0, 1});
            }
            {
                StreamScope scope(consumer, asynchronous, true);
                check(permute(value, {1, 2, 0}).raw_data() == bytes, "Storage layout lost bits or its cross-stream producer");
            }
        }
        producer.synchronize(); consumer.synchronize();
    }
}
}
int main(int argc, char** argv) {
    try {
        const Backend backend = parse_backend(argc > 1 ? argv[1] : "cpu");
        if (!backend_available(backend)) return 77;
        const size_t before = memory_stats(backend).live_bytes();
        malformed(backend); execution(backend); integer_constructors(backend);
        check(memory_stats(backend).live_bytes() == before, "Storage layout leaked allocation ownership");
        std::cout << "Storage layout: " << checks << " passed, 0 failed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
