#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include "Core/TensorDispatch.h"
#include "Inference/LLM/PagedAttention.h"
#include "IO/Json.h"
#include "Backends/MetalDTypeSupport.h"
#include "OperatorFixture.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string_view>

namespace {
using namespace elman;
using fixture::shape;

const std::set<std::string> registered = {
    // These operators are routed by operand dtype before table lookup.
    "angle", "complex", "conj", "float64_fma", "imag", "real", "sgn", "tanh",
#define ELMAN_CPU_KERNEL(operation) #operation,
#define ELMAN_CPU_KERNEL4(operation) #operation,
#include "Operators/cpu_dispatch.inc"
#undef ELMAN_CPU_KERNEL4
#undef ELMAN_CPU_KERNEL
};

const std::set<std::string> metal_kernels = {
#define ELMAN_METAL_KERNEL(operation) #operation,
#include "Operators/metal_dispatch.inc"
#undef ELMAN_METAL_KERNEL
};

Tensor run_case(const Json& test, const std::string& operation, Backend backend) {
    std::vector<Tensor> inputs;
    for (const auto& spec : test.at("inputs").array()) {
        inputs.push_back(fixture::input(spec, backend));
    }
    const auto& spec = test.at("output");
    const auto dimensions = shape(spec.at("shape"));
    const auto type = DType(spec.at("type").integer());
    auto result = dispatch_tensor(operation, dimensions, inputs, shape(test.at("attrs")), type, backend);
    if (result.shape() != dimensions || result.dtype() != type) throw std::runtime_error("Output descriptor differs");
    return result;
}

template<class Operation>
void check_rejection(const std::string& expected, Operation operation) {
    try { operation(); }
    catch (const std::invalid_argument& error) {
        if (error.what() == expected) return;
        throw;
    }
    throw std::runtime_error("Expected rejection: " + expected);
}

void check_metal_dtype_contract(const Json& test, const std::string& op) {
    KernelParams p;
    p.v[KernelParams::kOutputDTypeSlot] = int(test.at("output").at("type").integer());
    const auto& inputs = test.at("inputs").array();
    for (int i = 0; i < 4; ++i) {
        const auto slot = static_cast<KernelInput>(i);
        const bool present = size_t(i) < inputs.size() && !inputs[i].is_null();
        p.v[KernelParams::input_slot(slot)] = present ? int(inputs[i].at("shape").array().size()) : -1;
        if (present) p.v[KernelParams::input_dtype_slot(slot)] = int(inputs[i].at("type").integer());
    }
    auto validate = [&] { metal::validate_operator_dtypes(op,p); };
    if (test.contains("metal_error")) check_rejection(test.at("metal_error").string(),validate);
    else {
        validate();
        if (!metal_kernels.count(op)) throw std::runtime_error("Missing Metal kernel for " + op);
    }
}

void compare(const Tensor& expected, const Tensor& result, bool complex_math = false) {
    const auto dtype = expected.dtype();
    if (complex_math && is_complex(dtype)) {
        const auto actual = result.complex_data(), wanted = expected.complex_data();
        const double tolerance = dtype == DType::Complex32 ? .003 : 3e-5;
        for (size_t i = 0; i < wanted.size(); ++i) {
            if (actual[i] == wanted[i]) continue;
            if (!(std::abs(actual[i]-wanted[i]) <= tolerance * (1. + std::abs(wanted[i]))))
                throw std::runtime_error("Complex output mismatch at element " + std::to_string(i));
        }
        return;
    }
    if (!is_compute_float(dtype)) {
        if (result.raw_data() != expected.raw_data()) throw std::runtime_error("Output bytes differ");
        return;
    }
    const auto actual = result.data();
    const auto wanted = expected.data();
    const float tolerance = dtype == DType::Float32 ? 3e-4f : dtype == DType::Float16 ? .003f : .012f;
    for (size_t i = 0; i < wanted.size(); ++i) {
        const float a = actual[i], b = wanted[i];
        if (a == b || (std::isnan(a) && std::isnan(b))) continue;
        if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > tolerance * (1.f + std::abs(b)))
            throw std::runtime_error("Output mismatch at element " + std::to_string(i) + ": " +
                                     std::to_string(a) + " vs " + std::to_string(b));
    }
}

void check_distribution(const Tensor& values, double mean, double variance) {
    double sum = 0, squares = 0;
    const auto actual = values.data();
    for (float x : actual) {
        if (!std::isfinite(x)) throw std::runtime_error("Random operator produced nonfinite data");
        sum += x;
        squares += double(x)*x;
    }
    const double average = sum / actual.size();
    if (std::abs(average - mean) > .05 || std::abs(squares / actual.size() - average*average - variance) > .08)
        throw std::runtime_error("Random distribution differs");
}

void seeded_streams(Backend backend) {
    for (const auto type : {DType::Float32,DType::Float16,DType::BFloat16}) {
        const Shape dims{257};
        auto x = Tensor::ones(dims,backend,true,type);
        for (const std::string_view op : {"uniform", "normal", "dropout"}) {
            auto sample = [&](uint32_t seed) {
                if (op == "uniform") return Tensor::uniform(dims,-1.,1.,seed,backend,false,type);
                if (op == "normal") return Tensor::normal(dims,.5,.25,seed,backend,false,type);
                return dropout(x,.25,seed);
            };
            auto first = sample(17);
            auto second = sample(0x80000011u);
            auto replay = sample(17);
            if (first.raw_data() == second.raw_data() || first.raw_data() != replay.raw_data())
                throw std::runtime_error(std::string(op) + " failed distinct-seed or replay check");
            if (op == "dropout") {
                first.backward(Tensor::ones(dims,backend,false,type));
                if (x.grad() != first.data()) throw std::runtime_error("Dropout backward did not replay the forward mask");
            }
        }
    }
    auto tight = Tensor::uniform({33},1.,1.+1e-12,7,backend);
    compare(Tensor::ones({33}),tight);
    auto stream = [&](int offset) {
        return dispatch_tensor("uniform",{257},{},{17,offset,0,1065353216},DType::Float32,backend);
    };
    auto first = stream(0).raw_data();
    if (first == stream(1).raw_data() || first != stream(0).raw_data())
        throw std::runtime_error("Uniform stream offset did not select a reproducible sequence");
}

Tensor input(Shape dims, int seed, Backend device) {
    std::vector<float> data(shape_element_count(dims));
    for (size_t i = 0; i < data.size(); ++i)
        data[i] = float(int((i * 73 + i / 107 + seed) % 103) - 51) / 53.f;
    return Tensor::from_data(dims,data,device);
}

void decode_operator_paths(Backend backend) {
    if (backend != Backend::Metal) {
        for (auto dtype : {DType::Float32, DType::Float16, DType::BFloat16}) {
            auto gate = cast(input({2,257},9,backend),dtype);
            auto up = cast(input({257},21,backend),dtype);
            if (silu_mul(gate,expand(up,gate.shape())).raw_data() != silu_mul(gate,up).raw_data())
                throw std::runtime_error("Contiguous SiLU product differs from broadcasting");

            auto scores = cast(input({2,4,3,263},30,backend),dtype);
            auto probabilities = [&](const Tensor& pad) {
                return attention_softmax(scores,pad,11.3,3);
            };
            for (int batches : {1,2}) {
                auto pad = cast(input({batches,1,1,263},40,backend),dtype);
                if (probabilities(pad).raw_data() != probabilities(expand(pad,scores.shape())).raw_data())
                    throw std::runtime_error("Compact attention softmax mask differs from broadcasting");
            }

            auto q = cast(input({2,4,1,32},3,backend),dtype);
            auto k = cast(input({2,2,263,32},7,backend),dtype);
            auto v = cast(input({2,2,263,32},13,backend),dtype);
            auto pad = cast(input({2,1,1,263},40,backend),dtype);
            if (tiled_attention(q,k,v,pad,8,32,true,241).raw_data() !=
                tiled_attention(q,k,v,expand(pad,{2,4,1,263}),8,32,true,241).raw_data())
                throw std::runtime_error("Compact causal attention mask differs from broadcasting");
        }
    }

    std::vector<int32_t> slots(2*263), visible(slots.size());
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i] = i % 11 == 0 ? -1 : int32_t((i * 13) % 271);
        visible[i] = slots[i] >= 0 && i % 5 != 0;
    }
    auto paged_decode = [&](Backend device, DType dtype, bool single_query) {
        PagedAttentionMap pages(2,271,slots,visible,device);
        auto q = cast(input({2,4,1,128},3,device),dtype);
        if (!single_query) q = repeat(q,{1,1,2,1});
        auto result = paged_attention(q, cast(input({271,256},7,device),dtype),
                                       cast(input({271,256},13,device),dtype),pages,2,11.3f,1,true,250);
        return single_query ? result : slice(result,2,0,1);
    };
    for (auto dtype : {DType::Float16, DType::BFloat16}) {
        auto reference = paged_decode(Backend::CPU,dtype,false);
        auto decoded = paged_decode(backend,dtype,true);
        if (backend == Backend::CPU) {
            if (reference.raw_data() != decoded.raw_data())
                throw std::runtime_error("Long paged decode differs from row-by-row attention");
        } else {
            compare(reference,decoded);
        }
    }
}

void public_operators(Backend backend) {
    const std::vector<float> values = {1.f, -2.f, 3.f, 4.f, -.5f, 1.5f};
    auto reference = Tensor::from_data({2,3}, values, Backend::CPU, true);
    auto gpu = Tensor::from_data({2,3}, values, backend, true);
    auto forward = [](const Tensor& x) { return mean_all(silu(add(mul(x, .5), .25))); };
    auto cpu_loss = forward(reference), device_loss = forward(gpu);
    cpu_loss.backward();
    device_loss.backward();
    compare(cpu_loss, device_loss);
    compare(Tensor::from_data({2,3}, reference.grad()), Tensor::from_data({2,3}, gpu.grad()));

    AllocationPool pool(backend, 0);
    ExecutionStream stream(backend);
    StreamScope scope(stream);
    auto base = Tensor::from_data({2,3},values,backend);
    auto view = contiguous_view(base,{2,2},1);
    auto replay = add(view, 1);
    auto completed = stream.record();
    completed.synchronize();
    compare(Tensor::from_data({2,2},{-1.f,4.f,5.f,.5f}),replay);
    auto zero = Tensor::zeros({2,3},backend);
    compare(Tensor::zeros({2,3}),zero);
    for (auto type : {DType::Float32, DType::Float16, DType::BFloat16}) {
        auto x = Tensor::from_data({3},{-1,0,2},Backend::CPU,false,type);
        compare(angle(x),angle(x.to(backend)));
    }
    if (backend == Backend::Metal) {
        compare(mul(reference,-1.f),dispatch_tensor("neg",{2,3},{gpu},{},DType::Float32,backend));
        auto x = Tensor::from_double_data({1},{1.},backend);
        check_rejection("Metal does not support float64 arithmetic for add",[&] { add(x,x); });
        auto z = complex(x,x);
        check_rejection("Metal does not support complex128 arithmetic for mul",[&] { mul(z,z); });
    }

    auto grouped = [&](Backend device, int tile) {
        auto q = input({2,4,37,32},0,device);
        auto k = input({2,2,69,32},13,device);
        auto v = input({2,2,69,48},24,device);
        return tiled_attention(q,k,v,input({2,1,37,69},40,device),8,tile,true,-3);
    };
    for (int tile : {32, 64}) compare(grouped(Backend::CPU,tile),grouped(backend,tile));
    auto paged = [&](Backend device) {
        std::vector<int32_t> slots(2*73), visible(slots.size());
        for (size_t i = 0; i < slots.size(); ++i) {
            slots[i] = i % 11 == 0 ? -1 : int32_t((i * 13) % 71);
            visible[i] = slots[i] >= 0 && i % 5 != 0;
        }
        PagedAttentionMap pages(2,71,slots,visible,device);
        return paged_attention(input({2,4,5,64},3,device), input({71,128},7,device),
                               input({71,128},13,device), pages,2,8.f,3,true,4);
    };
    compare(paged(Backend::CPU),paged(backend));
    if (backend == Backend::CUDA) {
        // A partial reduction tile and a nonfinite upstream value at a padded corner.
        constexpr int height = 31, width = 35, inputs = 8, outputs = 64;
        std::vector<float> gradient(outputs * height * width, 1.f);
        gradient.front() = std::numeric_limits<float>::infinity();
        const Shape shape{outputs,inputs,3,3};
        auto actual = dispatch_tensor("conv2d_weight_grad",shape,
            {Tensor::ones({1,inputs,height,width},backend),Tensor::from_data({1,outputs,height,width},gradient,backend)},
            {1,1,1,1,1,1,1},DType::Float32,backend);
        std::vector<float> expected(shape_element_count(shape));
        for (size_t i = 0; i < expected.size(); ++i) {
            const int dy = int(i / 3 % 3) - 1, dx = int(i % 3) - 1;
            expected[i] = i < inputs * 9 && dy >= 0 && dx >= 0 ? gradient.front() :
                          float((height - std::abs(dy)) * (width - std::abs(dx)));
        }
        compare(Tensor::from_data(shape,expected),actual);
    }
    for (const Shape& dims : {Shape{2,97,3}, Shape{3,9,67}}) {
        auto cpu = softmax(input(dims,13,Backend::CPU),1);
        compare(cpu,softmax(input(dims,13,backend),1));
    }
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) throw std::invalid_argument("Usage: elman_operator_parity <fixtures.json> cpu|cuda|metal");
        const auto target = parse_backend(argv[2]);
        if (!backend_available(target)) return 77;
        const auto fixtures = Json::read(argv[1]);
        std::set<std::string> covered;
        size_t passed = 0, rejected = 0;
        int failures = 0;
        for (const auto& test : fixtures.array()) {
            const auto op = test.at("op").string();
            try {
                if (target == Backend::CPU) check_metal_dtype_contract(test,op);
                if (test.contains("error")) {
                    const auto error = test.at("error").string();
                    check_rejection(error,[&] { run_case(test,op,Backend::CPU); });
                    if (target != Backend::CPU) check_rejection(error,[&] { run_case(test,op,target); });
                    ++rejected;
                } else {
                    const auto reference = run_case(test,op,Backend::CPU);
                    if (target == Backend::Metal && test.contains("metal_error")) {
                        check_rejection(test.at("metal_error").string(),[&] { run_case(test,op,target); });
                        ++rejected;
                    } else if (target != Backend::CPU) {
                        compare(reference,run_case(test,op,target),test.bool_or("complex_math",false));
                    }
                    covered.insert(op);
                }
                ++passed;
            } catch (const std::exception& error) {
                ++failures;
                std::cerr << test.at("name").string() << ": " << error.what() << '\n';
            }
        }
        for (const auto& expected : registered) {
            if (!covered.count(expected)) {
                ++failures;
                std::cerr << "No passing fixture for " << expected << '\n';
            }
        }
        const std::set<Backend> devices{Backend::CPU, target};
        for (auto backend : devices) {
            check_distribution(Tensor::uniform({32768}, -1., 1., 17, backend), 0., 1./3.);
            check_distribution(Tensor::normal({32768}, 1., .5, 19, backend), 1., .25);
            auto x = Tensor::ones({32768}, backend);
            auto dropped = dropout(x, .2, 21);
            check_distribution(dropped, 1., .25);
            compare(dropped,dropout(x,.2,21));
            seeded_streams(backend);
        }
        if (target != Backend::CPU) public_operators(target);
        decode_operator_paths(target);
        std::cout << backend_name(target) << " passed " << passed << '/' << fixtures.array().size()
                  << " fixtures covering " << covered.size() << '/' << registered.size() << " native operators, "
                  << rejected << " expected rejections\n";
        return failures ? 1 : 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
