#include "Operators/recurrent_ops.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(const Tensor& actual, const std::vector<float>& expected, float tolerance = 2e-5f) {
    const auto data = cast(actual, DType::Float32).data();
    check(data.size() == expected.size(), "Recurrent result size differs");
    for (size_t i = 0; i < data.size(); ++i)
        check(std::isfinite(data[i]) && std::abs(data[i] - expected[i]) < tolerance, "Recurrent result differs from scalar reference");
}
std::vector<float> values(int count, double phase) {
    std::vector<float> result(count);
    for (int i = 0; i < count; ++i) result[i] = float(.3 * std::sin(i * .47 + phase));
    return result;
}
void recurrence(Backend backend, DType dtype) {
    constexpr int B = 2, T = 37, H = 4, G = 2, K = 3, V = 5;
    const auto q = Tensor::from_data({B,T,G,K}, values(B*T*G*K, 0), backend, false, dtype);
    const auto k = Tensor::from_data({B,T,G,K}, values(B*T*G*K, 1), backend, false, dtype);
    const auto v = Tensor::from_data({B,T,H,V}, values(B*T*H*V, 2), backend, false, dtype);
    const auto g = Tensor::from_data({B,T,H}, std::vector<float>(B*T*H, -.17f), backend);
    const auto beta = Tensor::from_data({B,T,H}, std::vector<float>(B*T*H, .7f), backend);
    const auto state = Tensor::from_data({B,H,K,V}, values(B*H*K*V, 3), backend);
    const auto qv = cast(q, DType::Float32).data(), kv = cast(k, DType::Float32).data(), vv = cast(v, DType::Float32).data();
    const auto state_values = state.data();
    std::vector<double> memory(state_values.begin(), state_values.end());
    std::vector<float> expected(B*T*H*V);
    for (int b = 0; b < B; ++b) for (int t = 0; t < T; ++t) for (int h = 0; h < H; ++h) {
        const int base = ((b*T+t)*G+h/(H/G))*K;
        double qnorm = 1e-6, knorm = 1e-6;
        for (int d = 0; d < K; ++d) { qnorm += double(qv[base+d])*qv[base+d]; knorm += double(kv[base+d])*kv[base+d]; }
        for (int column = 0; column < V; ++column) {
            double prediction = 0;
            for (int d = 0; d < K; ++d) {
                auto& item = memory[((b*H+h)*K+d)*V+column];
                item *= std::exp(-.17f); prediction += item * kv[base+d] / std::sqrt(knorm);
            }
            const int target = ((b*T+t)*H+h)*V+column;
            const double delta = (vv[target] - prediction) * .7f;
            double output = 0;
            for (int d = 0; d < K; ++d) {
                auto& item = memory[((b*H+h)*K+d)*V+column];
                item += kv[base+d] / std::sqrt(knorm) * delta;
                output += item * qv[base+d] / std::sqrt(qnorm*K);
            }
            expected[target] = float(output);
        }
    }
    const auto full = gated_delta_rule(q,k,v,g,beta,state);
    near(full.values, expected, dtype == DType::Float16 ? .0003f : 2e-5f);
    std::vector<float> expected_state(memory.size());
    std::transform(memory.begin(), memory.end(), expected_state.begin(), [](double value) { return float(value); });
    near(full.state, expected_state);
    const auto first = gated_delta_rule(slice(q,1,0,13),slice(k,1,0,13),slice(v,1,0,13),slice(g,1,0,13),slice(beta,1,0,13),state);
    const auto last = gated_delta_rule(slice(q,1,13,T),slice(k,1,13,T),slice(v,1,13,T),slice(g,1,13,T),slice(beta,1,13,T),first.state);
    near(concat({first.values,last.values},1), cast(full.values,DType::Float32).data());
    near(last.state, full.state.data()); near(state,state_values);
}
}
int main(int argc, char** argv) {
    try {
        check(argc == 2, "Usage: elman_recurrent_tests BACKEND");
        const auto backend = parse_backend(argv[1]); if (!backend_available(backend)) return 77;
        recurrence(backend,DType::Float32); recurrence(backend,DType::Float16);
        const auto input = Tensor::from_data({1,4,2},{1,10,2,20,3,30,4,40},backend);
        const auto weights = Tensor::from_data({2,3},{1,2,3,.1f,.2f,.3f},backend);
        const auto full = causal_depthwise_conv1d(input,weights);
        near(full.values,{3,3,8,8,14,14,20,20}); near(full.state,{3,30,4,40});
        const auto first = causal_depthwise_conv1d(slice(input,1,0,1),weights);
        const auto rest = causal_depthwise_conv1d(slice(input,1,1,4),weights,first.state);
        near(concat({first.values,rest.values},1),full.values.data());
        near(causal_depthwise_conv1d(input,Tensor::ones({2,1},backend)).values,input.data());
        bool rejected = false;
        try { causal_depthwise_conv1d(input,weights,Tensor::zeros({1,3,2},backend)); }
        catch (const std::invalid_argument&) { rejected = true; }
        check(rejected,"Invalid convolution state accepted");
        std::cout << "Recurrent operator checks passed\n"; return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
