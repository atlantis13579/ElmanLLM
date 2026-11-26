#include "Core/Execution.h"
#include "Core/Tensor.h"
#include "Core/TensorDispatch.h"
#include "Fusion/Fusion.h"
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>

using namespace elman;
namespace {
fusion::Options options(bool enabled) {
    fusion::Options result;
    result.enabled = enabled;
    result.cache_directory = "Intermediate/MetalFusionTests";
    return result;
}
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void near(const std::vector<float>& a, const std::vector<float>& b) {
    check(a.size() == b.size(), "Metal fusion result size mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        check(std::isfinite(a[i]) && std::isfinite(b[i]) &&
              std::abs(double(a[i]) - b[i]) <= 1e-4 + 1e-4 * std::abs(double(a[i])),
              "Metal fusion numerical mismatch");
}
std::vector<float> data(int count, int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> distribution(-.5f, .5f);
    std::vector<float> result(count);
    for (float& value : result) value = distribution(generator);
    return result;
}
std::vector<std::vector<float>> expression(bool enabled, int width, int seed) {
    fusion::Policy policy(options(enabled));
    auto x = Tensor::from_data({3, width}, data(3 * width, seed), Backend::Metal, true);
    auto bias = Tensor::from_data({width}, data(width, seed + 1), Backend::Metal, true);
    auto dy = Tensor::from_data(x.shape(), data(3 * width, seed + 2), Backend::Metal);
    fusion::Scope scope;
    auto saved = add(x, bias);
    auto centered = sub(saved, mean(saved, -1, true));
    auto normalized = mul(centered, rsqrt(add(mean(mul(centered, centered), -1, true), .01)));
    auto probability = softmax(gelu(normalized), -1);
    auto output = add(dropout(probability, .2, 1337, true), saved);
    output.backward(dy);
    scope.finish();
    return {saved.data(), output.data(), x.grad(), bias.grad()};
}
void boundaries() {
    fusion::Policy policy(options(true));
    auto x = Tensor::from_data({37}, data(37, 42), Backend::Metal);
    const auto input = x.data();
    Tensor output, view;
    {
        fusion::Scope scope;
        auto saved = add(x, 1.);
        view = contiguous_view(saved, {17}, 1);
        output = mul(saved, 2.);
        saved = {};
        // Mutating a feed must materialize all preceding reads first.
        x.assign(std::vector<float>(37, 7.f));
        scope.finish();
    }
    std::vector<float> expected = input, expected_view(input.begin() + 1, input.begin() + 18);
    for (float& value : expected) value = (value + 1.f) * 2.f;
    for (float& value : expected_view) value += 1.f;
    near(expected, output.data());
    near(expected_view, view.data());
    {
        fusion::Scope scope;
        auto pending = mul(add(x, 1.), .5);
        near(std::vector<float>(37, 4.f), pending.data());
        dispatch_tensor_into("add", pending, {pending, x});
        scope.finish();
        near(std::vector<float>(37, 11.f), pending.data());
    }
    ExecutionStream first(Backend::Metal), second(Backend::Metal);
    {
        fusion::Scope scope;
        { StreamScope stream(first, true); output = mul(add(x, 1.), 2.); }
        { StreamScope stream(second, true); output = sub(mul(output, 3.), 4.); second.record().synchronize(); }
        scope.finish();
    }
    near(std::vector<float>(37, 44.f), output.data());
}
std::vector<float> training(bool enabled) {
    fusion::Policy policy(options(enabled));
    auto weight = Tensor::from_data({17}, data(17, 99), Backend::Metal, true);
    std::vector<float> losses;
    for (int step = 0; step < 6; ++step) {
        fusion::Scope scope;
        auto error = sub(mul(weight, 2.), .25);
        auto loss = mean(mul(error, error), -1, false);
        loss.backward();
        scope.finish();
        losses.push_back(loss.data().at(0));
        auto values = weight.data(), gradient = weight.grad();
        for (size_t i = 0; i < values.size(); ++i) values[i] -= .1f * gradient[i];
        weight.assign(values);
        weight.zero_grad();
    }
    check(losses.back() < losses.front(), "Metal training loss must decrease");
    return losses;
}
} // namespace

int main() {
    try {
        if (!backend_available(Backend::Metal)) return 77;
        fusion::set_enabled(true);
        fusion::reset_statistics();
        // A second input seed exercises already cached graphs with new buffers
        // and values. Width 1025 exercises the conservative reduction boundary.
        for (int width : {1, 17, 384, 1025}) for (int seed : {42, 73}) {
            const auto reference = expression(false, width, seed);
            const auto actual = expression(true, width, seed);
            for (size_t i = 0; i < reference.size(); ++i) near(reference[i], actual[i]);
        }
        boundaries();
        near(training(false), training(true));
        const auto stats = fusion::statistics();
        check(stats.launches > 0 && stats.verified > 0, "Metal fusion programs must execute and verify");
        check(stats.fused_commands > stats.launches, "Metal regions must combine dispatches");
        check(stats.compile_failures == 0, "Metal fusion graph compilation must succeed");
        fusion::set_enabled(false);
        const auto launches = fusion::statistics().launches;
        training(true);
        check(fusion::statistics().launches == launches, "Global fusion-off must override local policy");
        fusion::set_enabled(true);
        std::cout << "PASS Metal fusion: compiled=" << stats.compiled << " launches=" << stats.launches
                  << " verified=" << stats.verified << " rejected=" << stats.rejected << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
