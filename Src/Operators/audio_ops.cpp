#include "Operators/audio_ops.h"
#include "Core/Device.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace elman {
namespace {
constexpr double pi = 3.14159265358979323846;
void require(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
double mel(double hz) {
    return hz < 1000 ? hz * 3 / 200 : 15 + std::log(hz / 1000) * 27 / std::log(6.4);
}
double hz(double value) {
    return value < 15 ? value * 200 / 3 : 1000 * std::exp((value - 15) * std::log(6.4) / 27);
}
}

Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias,
              int stride, int padding, int dilation) {
    require(input.shape().size() == 3 && weight.shape().size() == 3,
            "Conv1d requires NCT input and OCK weights");
    const auto& x = input.shape();
    const auto& w = weight.shape();
    require(is_compute_float(input.dtype()) && input.dtype() == weight.dtype() &&
                input.backend() == weight.backend() && input.device() == weight.device(),
            "Conv1d input and weight must share a floating dtype and device");
    require(x[0] > 0 && x[1] > 0 && x[2] > 0 && w[0] > 0 && w[1] == x[1] && w[2] > 0 &&
                stride > 0 && padding >= 0 && dilation > 0, "Invalid conv1d dimensions");
    const int64_t extent = int64_t(x[2]) + 2LL * padding - int64_t(dilation) * (w[2] - 1) - 1;
    require(extent >= 0 && extent / stride < std::numeric_limits<int>::max(), "Invalid conv1d output length");
    const int length = static_cast<int>(extent / stride + 1);
    require(int64_t(length) * w[2] <= std::numeric_limits<int>::max() &&
                int64_t(w[1]) * w[2] <= std::numeric_limits<int>::max(), "Conv1d unfolding is too large");
    require(!bias.defined() || (bias.shape() == Shape{w[0]} && bias.dtype() == input.dtype() &&
                bias.backend() == input.backend() && bias.device() == input.device()), "Invalid conv1d bias");
    DeviceScope owner(input.backend(), input.device());
    std::vector<int32_t> positions(size_t(length) * w[2]);
    for (int t = 0; t < length; ++t) for (int k = 0; k < w[2]; ++k) {
        const int64_t source = int64_t(t) * stride + int64_t(k) * dilation - padding;
        positions[size_t(t) * w[2] + k] = source < 0 || source >= x[2] ? x[2] : int32_t(source);
    }
    const Tensor zero = Tensor::zeros({x[0], x[1], 1}, input.backend(), false, input.dtype());
    Tensor patches = index_select(concat({input, zero}, 2), 2,
                                  Tensor::from_int_data({int(positions.size())}, positions, input.backend()));
    patches = reshape(permute(reshape(patches, {x[0], x[1], length, w[2]}), {0, 2, 1, 3}),
                      {x[0], length, w[1] * w[2]});
    return transpose(linear(patches, reshape(weight, {w[0], w[1] * w[2]}), bias), 1, 2);
}

LogMelSpectrogram::LogMelSpectrogram(Backend backend, int mel_bins, int sample_rate, int fft_size, int hop_length)
    : backend_(backend), mel_bins_(mel_bins), fft_size_(fft_size), hop_length_(hop_length) {
    require(mel_bins > 0 && mel_bins <= 512 && sample_rate > 0 && fft_size >= 4 && fft_size <= 4096 &&
                fft_size % 2 == 0 && hop_length > 0, "Invalid log-mel dimensions");
    const int bins = fft_size / 2 + 1;
    std::vector<float> window(fft_size), fourier(size_t(2 * bins) * fft_size);
    for (int i = 0; i < fft_size; ++i) {
        window[i] = float(.5 - .5 * std::cos(2 * pi * i / fft_size));
        for (int f = 0; f < bins; ++f) {
            const double phase = 2 * pi * f * i / fft_size;
            fourier[size_t(f) * fft_size + i] = float(std::cos(phase));
            fourier[size_t(f + bins) * fft_size + i] = float(-std::sin(phase));
        }
    }
    std::vector<double> edges(mel_bins + 2);
    for (int i = 0; i < mel_bins + 2; ++i) edges[i] = hz(mel(sample_rate / 2.) * i / (mel_bins + 1));
    std::vector<float> filters(size_t(mel_bins) * bins);
    for (int m = 0; m < mel_bins; ++m) for (int f = 0; f < bins; ++f) {
        const double frequency = double(f) * sample_rate / fft_size;
        const double rising = (frequency - edges[m]) / (edges[m + 1] - edges[m]);
        const double falling = (edges[m + 2] - frequency) / (edges[m + 2] - edges[m + 1]);
        filters[size_t(m) * bins + f] = float(std::max(0., std::min(rising, falling)) * 2 / (edges[m + 2] - edges[m]));
    }
    window_ = Tensor::from_data({fft_size}, window, backend);
    fourier_ = Tensor::from_data({2 * bins, fft_size}, fourier, backend);
    filters_ = Tensor::from_data({mel_bins, bins}, filters, backend);
}

Tensor LogMelSpectrogram::operator()(const Tensor& samples) const {
    require(samples.shape().size() == 1 && samples.shape()[0] > fft_size_ / 2 &&
                samples.dtype() == DType::Float32 && samples.backend() == backend_ &&
                samples.device() == window_.device(), "Log-mel input must be a floating waveform on the filterbank device");
    const int count = samples.shape()[0], frames = count / hop_length_, bins = fft_size_ / 2 + 1;
    DeviceScope owner(samples.backend(), samples.device());
    require(frames > 0 && int64_t(frames) * fft_size_ <= std::numeric_limits<int>::max(),
            "Log-mel input duration is out of range");
    std::vector<int32_t> indices(size_t(frames) * fft_size_);
    for (int t = 0; t < frames; ++t) for (int i = 0; i < fft_size_; ++i) {
        int index = t * hop_length_ + i - fft_size_ / 2;
        if (index < 0) index = -index;
        if (index >= count) index = 2 * count - 2 - index;
        indices[size_t(t) * fft_size_ + i] = index;
    }
    Tensor framed = reshape(index_select(samples, 0,
        Tensor::from_int_data({int(indices.size())}, indices, backend_)), {frames, fft_size_});
    Tensor spectrum = linear(mul(framed, window_), fourier_);
    Tensor real_part = slice(spectrum, 1, 0, bins), imaginary_part = slice(spectrum, 1, bins, 2 * bins);
    Tensor power = transpose(add(mul(real_part, real_part), mul(imaginary_part, imaginary_part)), 0, 1);
    Tensor energies = matmul(filters_, power);
    Tensor log_spec = mul(log(maximum(energies, Tensor::from_data({}, {1e-10f}, backend_))), 1 / std::log(10.));
    Tensor peak = amax(reshape(log_spec, {int(log_spec.numel())}), 0);
    return mul(add(maximum(log_spec, sub(peak, 8)), 4), .25);
}

} // namespace elman
