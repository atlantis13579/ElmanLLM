#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Core/Graph.h"
#include "Core/TensorDispatch.h"
#include <algorithm>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {
using namespace elman;
int checks = 0;
void check(bool value, const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F action, const char* message) {
    bool rejected = false;
    try { action(); } catch (const std::invalid_argument&) { rejected = true; }
    check(rejected, message);
}
Tensor indices(Backend backend, const std::vector<int32_t>& values, bool wide) {
    if (!wide) return Tensor::from_int_data({int(values.size())}, values, backend);
    std::vector<int64_t> data(values.begin(), values.end());
    std::vector<uint8_t> bytes(data.size()*sizeof(int64_t));
    if (!bytes.empty()) std::memcpy(bytes.data(), data.data(), bytes.size());
    return Tensor::from_raw({int(values.size())}, DType::Int64, bytes, backend);
}
void mutation(Backend backend, bool pooled) {
    size_t before = memory_stats(backend).live_bytes();
    {
        AllocationPool pool(backend, 0, pooled);
        for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16, DType::Int32, DType::Bool}) {
            for (bool wide : {false, true}) {
                Tensor root = Tensor::zeros({32}, backend, false, dtype);
                Tensor cache = contiguous_view(root, {5, 3}, 3);
                Tensor source = Tensor::ones({3, 3}, backend, false, dtype);
                Tensor ids = indices(backend, {4, 0, 2}, wide);
                void* pointer = cache.node()->value->raw_pointer();
                uint64_t version = root.node()->value->version, requests = pool.stats().requests;
                cache.assign_indexed(0, ids, source);
                check(cache.node()->value->raw_pointer() == pointer, "Indexed mutation replaced storage");
                check(pool.stats().requests == requests, "Unique indexed mutation allocated tensor/workspace storage");
                check(root.node()->value->version == version + 1, "Indexed mutation lost the shared version");
                std::vector<float> expected(32);
                for (int row : {4, 0, 2}) std::fill_n(expected.begin() + 3 + row*3, 3, 1.0f);
                check(root.data() == expected, "Indexed mutation escaped its output view");
                std::vector<uint8_t> saved = root.raw_data();
                version = root.node()->value->version;
                Tensor invalid = indices(backend, {0, 5, 2}, wide);
                rejects([&] { cache.assign_indexed(0, invalid, source); }, "Invalid mapping was accepted");
                check(root.raw_data() == saved && root.node()->value->version == version, "Rejected mapping mutated storage/version");
                Tensor empty_ids = indices(backend, {}, wide);
                cache.assign_indexed(0, empty_ids, Tensor::zeros({0, 3}, backend, false, dtype));
                check(root.raw_data() == saved && root.node()->value->version == version + 1, "Empty mutation changed data or lost its version");
                // Disjoint ranges sharing one allocation are valid and retain
                // the root's common read/write dependency state.
                Tensor disjoint = contiguous_view(root, {1, 3}, 24);
                disjoint.assign({1, 1, 1});
                cache.assign_indexed(0, indices(backend, {1}, wide), disjoint);
                check(index_select(cache, 0, indices(backend, {1}, false)).data() == std::vector<float>(3, 1), "Disjoint aliases failed");
                Tensor overlap = contiguous_view(root, {1, 3}, 4);
                rejects([&] { cache.assign_indexed(0, indices(backend, {0}, wide), overlap); }, "Overlapping source was accepted");
            }
        }
    }
    check(memory_stats(backend).live_bytes() == before, "Indexed mutation leaked storage");
}
void gradients(Backend backend) {
    for (elman::DType dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
        Tensor x = Tensor::ones({3, 2}, backend, true, dtype), source = Tensor::ones({3, 2}, backend, true, dtype);
        Tensor ids = indices(backend, {2, 0, 2}, true);
        Tensor result = index_copy(x, 0, ids, source);
        result.backward(Tensor::from_data({3, 2}, {1, 2, 3, 4, 5, 6}, backend, false, dtype));
        check(x.grad() == std::vector<float>{0, 0, 3, 4, 0, 0}, "Destination gradient differs");
        check(source.grad() == std::vector<float>{5, 6, 1, 2, 5, 6}, "Duplicate source gradient differs");
        Tensor plain = Tensor::ones({3, 2}, backend, false, dtype);
        rejects([&] { x.assign_indexed(0, ids, plain); }, "Gradient target mutation accepted");
        rejects([&] { plain.assign_indexed(0, ids, source); }, "Gradient source mutation accepted");
        for (int changed = 0; changed < 3; ++changed) {
            Tensor a = Tensor::ones({3, 2}, backend, true, dtype), b = Tensor::ones({3, 2}, backend, true, dtype);
            Tensor mapping = indices(backend, {2, 0, 2}, changed % 2 == 0);
            Tensor out = index_copy(a, 0, mapping, b);
            if (changed == 0) a.assign(std::vector<float>(6, 2));
            else if (changed == 1) b.assign(std::vector<float>(6, 2));
            else mapping.assign_raw(mapping.raw_data());
            rejects([&] { out.backward(Tensor::ones({3, 2}, backend, false, dtype)); }, "Saved indexed operand mutation escaped validation");
        }
        Tensor saved = Tensor::ones({3, 2}, backend, false, dtype);
        Tensor leaf = Tensor::ones({3, 2}, backend, true, dtype);
        Tensor out = mul(leaf, saved);
        saved.assign_indexed(0, ids, plain);
        rejects([&] { out.backward(Tensor::ones({3, 2}, backend, false, dtype)); }, "Indexed mutation escaped an existing graph");
    }
}
void contracts(Backend backend) {
    Tensor x = Tensor::zeros({3, 2}, backend), s = Tensor::ones({2, 2}, backend);
    Tensor ids = indices(backend, {2, 0}, false), out = Tensor::zeros({3, 2}, backend);
    KernelParams p;
    p.v[0] = 6; p.v[1] = 2; p.v[2] = 3; p.v[3] = 2;
    p.v[KernelParams::input_slot(KernelInput::D)] = -1;
    std::vector<BufferPtr> inputs;
    int i = 0;
    for (const elman::Tensor& t : {x, ids, s}) {
        KernelInput input = KernelInput(i++);
        int slot = KernelParams::input_slot(input), stride = 1;
        p.v[slot] = int(t.shape().size());
        p.v[KernelParams::input_dtype_slot(input)] = int(t.dtype());
        for (int axis = int(t.shape().size())-1; axis >= 0; --axis) {
            p.v[slot+1+axis] = t.shape()[axis]; p.v[slot+9+axis] = stride; stride *= t.shape()[axis];
        }
        inputs.push_back(t.node()->value);
    }
    backend_dispatch("index_copy", out.node()->value, inputs, p);
    check(out.data() == std::vector<float>{1, 1, 0, 0, 1, 1}, "Raw dispatch differs");
    for (std::pair<int, int> corruption : std::vector<std::pair<int, int>>{
            {0, 5}, {1, 9}, {2, 2}, {64, 2}, {64, -3}, {10, -1}, {10, 9},
            {11, -3}, {19, 1}, {27, 2}, {28, 3}, {36, 2}, {44, 1}, {46, 3}, {53, 9}, {253, 0}, {255, 1}}) {
        KernelParams bad = p; bad.v[corruption.first] = corruption.second;
        uint64_t dispatches = backend_dispatch_count(backend);
        rejects([&] { backend_dispatch("index_copy", out.node()->value, inputs, bad); }, "Malformed raw descriptor accepted");
        check(backend_dispatch_count(backend) == dispatches, "Malformed descriptor reached a kernel");
    }
    std::vector<BufferPtr> extra = inputs; extra.push_back(nullptr); extra.push_back(nullptr);
    rejects([&] { backend_dispatch("index_copy", out.node()->value, extra, p); }, "Extra operands accepted");
    rejects([&] { backend_dispatch("index_copy", out.node()->value, {inputs[0], inputs[1]}, p); }, "Missing source accepted");
    Tensor root = Tensor::zeros({12}, backend);
    Tensor base = contiguous_view(root, {3, 2}), partial = contiguous_view(root, {3, 2}, 2);
    rejects([&] { dispatch_tensor_into("index_copy", partial, {base, ids, s}, {0}); }, "Partial base/output overlap accepted");
    Tensor integer = Tensor::from_int_data({3}, {0, 1, 2}, backend);
    rejects([&] { integer.assign_indexed(0, contiguous_view(integer, {1}), Tensor::from_int_data({1}, {7}, backend)); },
            "Output/index overlap accepted");
    if (backend == Backend::CUDA) {
        rejects([&] { index_copy(x, 0, ids.to(Backend::CPU), s); }, "Mixed backend indices accepted");
        rejects([&] { index_copy(x, 0, ids, s.to(Backend::CPU)); }, "Mixed backend source accepted");
    }
}
void large_indices(Backend backend) {
    // A valid index at 2^24 must not be rounded together with the dimension.
    Tensor x = Tensor::zeros({16777217}, backend);
    Tensor last = indices(backend, {16777216}, false), bad = indices(backend, {16777217}, false);
    x.assign_indexed(0, last, Tensor::from_data({1}, {7}, backend));
    check(index_select(x, 0, last).data() == std::vector<float>{7}, "Valid large index_select index rejected/rounded");
    check(gather(x, 0, last).data() == std::vector<float>{7}, "Valid large gather index rejected/rounded");
    rejects([&] { index_select(x, 0, bad); }, "End index_select index accepted");
    rejects([&] { gather(x, 0, bad); }, "End gather index accepted");
    rejects([&] { x.assign_indexed(0, bad, Tensor::ones({1}, backend)); }, "End index_copy index accepted");
}
void streams(Backend backend, bool pooled) {
    size_t before = memory_stats(backend).live_bytes();
    {
        AllocationPool pool(backend, 0, pooled);
        ExecutionStream writer(backend), reader(backend);
        for (bool wide : {false, true}) for (int repeat = 0; repeat < 12; ++repeat) {
            Tensor root, cache, result, input, ids;
            {
                StreamScope scope(writer);
                root = Tensor::zeros({37, 129}, backend, false, DType::BFloat16);
                cache = contiguous_view(root, {33, 129}, 129);
                ids = indices(backend, {32, 0, 16, 0, 32}, wide);
                input = Tensor::from_data({5, 129}, std::vector<float>(5*129, float(repeat+1)), backend, false, DType::BFloat16);
                cache.assign_indexed(0, ids, input);
            }
            {
                StreamScope scope(reader);
                result = index_select(cache, 0, indices(backend, {0, 16, 32}, false));
            }
            {
                StreamScope scope(writer);
                cache.assign_indexed(0, ids, Tensor::zeros({5, 129}, backend, false, DType::BFloat16));
            }
            root = {}; cache = {}; input = {}; ids = {};
            check(result.data() == std::vector<float>(3*129, float(repeat+1)), "Cross-stream read/write or released-source ordering failed");
        }
        writer.synchronize(); reader.synchronize();
    }
    check(memory_stats(backend).live_bytes() == before, "Indexed stream workspace leaked");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_indexed_copy cpu|cuda");
        Backend backend = parse_backend(argv[1]);
        if (!backend_available(backend)) return 77;
        for (bool pooled : {false, true}) { mutation(backend, pooled); streams(backend, pooled); }
        gradients(backend); contracts(backend); large_indices(backend);
        std::cout << "Indexed copy " << argv[1] << ": " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
