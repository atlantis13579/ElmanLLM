#pragma once

#include "Core/Tensor.h"

namespace elman {

// NCT input and OCK weights. Unfolding feeds the selected backend's GEMM.
Tensor conv1d(const Tensor& input, const Tensor& weight, const Tensor& bias = {},
              int stride = 1, int padding = 0, int dilation = 1);

// Centered periodic-Hann STFT, Slaney mel filters and Whisper log compression.
// Windowing, Fourier projection and filterbank multiplication stay on the input device.
class LogMelSpectrogram {
public:
    LogMelSpectrogram(Backend backend, int mel_bins = 80, int sample_rate = 16000,
                       int fft_size = 400, int hop_length = 160);
    Tensor operator()(const Tensor& samples) const;
    int mel_bins() const { return mel_bins_; }
private:
    Backend backend_;
    int mel_bins_, fft_size_, hop_length_;
    Tensor window_, fourier_, filters_;
};

} // namespace elman
