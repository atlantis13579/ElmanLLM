#include "Operators/audio_ops.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace elman;
namespace {
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
void near(const Tensor& value, const std::vector<float>& expected, float tolerance = 1e-5f) {
    const auto actual = value.data();
    check(actual.size() == expected.size(), "Unexpected audio tensor size");
    for (size_t i = 0; i < actual.size(); ++i)
        check(std::abs(actual[i] - expected[i]) < tolerance, "Audio numerical mismatch");
}
template<class F> void rejects(F action) {
    bool rejected = false;
    try { action(); } catch (const std::exception&) { rejected = true; }
    check(rejected, "Invalid audio operator input was accepted");
}
}
int main(int argc, char** argv) {
    try {
        const Backend backend = parse_backend(argc > 1 ? argv[1] : "cpu");
        if (!backend_available(backend)) return 77;
        Tensor x = Tensor::from_data({1, 1, 3}, {1, 2, 3}, backend, true);
        Tensor w = Tensor::from_data({1, 1, 2}, {2, 3}, backend, true);
        const Tensor convolution = conv1d(x, w);
        near(convolution, {8, 13});
        sum_all(convolution).backward();
        check(x.grad() == std::vector<float>{2, 5, 3}, "Incorrect convolution input gradient");
        check(w.grad() == std::vector<float>{3, 5}, "Incorrect convolution weight gradient");
        near(conv1d(x.detach(), w.detach(), {}, 2, 1), {3, 13});
        near(conv1d(x.detach(), w.detach(), {}, 1, 0, 2), {11});
        near(conv1d(Tensor::from_data({2, 2, 2}, {1,2,3,4,5,6,7,8}, backend),
                    Tensor::from_data({1, 2, 1}, {2,3}, backend),
                    Tensor::from_data({1}, {1}, backend)), {12,17,32,37});
        rejects([&] { conv1d(x, w, {}, 0); });
        rejects([&] { conv1d(x, w, {}, 1, -1); });
        rejects([&] { conv1d(x, reshape(w, {2})); });
        LogMelSpectrogram filter(backend, 8, 16000, 32, 8);
        near(filter(Tensor::zeros({64}, backend)), std::vector<float>(64, -1.5f));
        std::vector<float> samples(64);
        for (int i = 0; i < 64; ++i) samples[i] = float((i * 7) % 19 - 9) / 10;
        // Reference: periodic-Hann torch.stft with reflect centering and a Slaney filterbank.
        const std::vector<float> expected{
            0.28443187f, 0.22512531f, 0.19712681f, 0.1757654f, 0.20419782f, 0.17365074f, 0.18993449f, 0.15650803f,
            0.29519653f, 0.23014277f, 0.19662875f, 0.1863274f, 0.20179367f, 0.18366945f, 0.19128519f, 0.15822846f,
            0.49761307f, 0.37973773f, 0.23728001f, 0.38729697f, 0.1380474f, 0.3806473f, 0.28220904f, 0.25683832f,
            0.38440424f, 0.40840125f, 0.38319945f, 0.43309939f, 0.36904532f, 0.42841935f, 0.39451414f, 0.39502561f,
            0.16553515f, 0.37414342f, 0.42688578f, 0.32760894f, 0.44029725f, 0.34256971f, 0.41417086f, 0.42201239f,
            0.27001953f, 0.3229636f, 0.31275767f, 0.3332544f, 0.30936795f, 0.32974643f, 0.31521446f, 0.30559999f,
            0.50981021f, 0.47165096f, 0.46081632f, 0.45864248f, 0.46128917f, 0.45906854f, 0.46051615f, 0.44605339f,
            0.54828906f, 0.55220592f, 0.55189288f, 0.55864131f, 0.55012709f, 0.55810475f, 0.55327868f, 0.55727446f,
        };
        near(filter(Tensor::from_data({64}, samples, backend)), expected, 2e-4f);
        rejects([&] { filter(Tensor::zeros({8}, backend)); });
        rejects([&] { LogMelSpectrogram invalid(backend, 8, 16000, 31, 8); });
        std::cout << "Audio component checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
