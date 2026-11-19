#include "IO/Jpeg.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace elman {
namespace {

using Block = std::array<double, 64>;

constexpr std::array<int, 64> zigzag{
   0, 1, 8,16, 9, 2, 3,10, 17,24,32,25,18,11, 4, 5,
  12,19,26,33,40,48,41,34, 27,20,13, 6, 7,14,21,28,
  35,42,49,56,57,50,43,36, 29,22,15,23,30,37,44,51,
  58,59,52,45,38,31,39,46, 53,60,61,54,47,55,62,63
};

// A fixed quantizer for 4:4:4, with finer steps at low frequencies.
constexpr std::array<int, 64> quant{
   4, 3, 3, 4, 6,10,13,15,
   3, 3, 4, 5, 7,14,15,13,
   4, 3, 4, 6,10,14,17,13,
   4, 4, 6, 8,13,21,19,15,
   5, 6, 9,13,16,26,25,18,
   6, 9,13,15,19,25,27,22,
  12,15,19,21,25,29,29,24,
  17,22,23,24,27,24,25,24
};

void append_byte(std::string& output, unsigned byte) {
  output.push_back(static_cast<char>(byte & 255));
}

void append_word(std::string& output, unsigned word) {
  append_byte(output, word >> 8);
  append_byte(output, word);
}

void marker(std::string& output, unsigned type) {
  append_byte(output, 0xff);
  append_byte(output, type);
}

void segment(std::string& output, unsigned type, std::string_view data) {
  marker(output, type);
  append_word(output, static_cast<unsigned>(data.size()) + 2);
  output.append(data);
}

// Canonical tables use one code length per table. DC codes represent
// categories 0 to 11. AC codes represent EOB, ZRL, then each run/size pair.
std::string huffman_tables() {
  std::string tables;
  append_byte(tables, 0x00);
  const auto dc = tables.size();
  tables.append(16, '\0');
  tables[dc + 3] = 12;
  for (unsigned size = 0; size <= 11; ++size) append_byte(tables, size);
  append_byte(tables, 0x10);
  const auto ac = tables.size();
  tables.append(16, '\0');
  tables[ac + 7] = static_cast<char>(162);
  append_byte(tables, 0);
  append_byte(tables, 0xf0);
  for (unsigned run = 0; run < 16; ++run)
    for (unsigned size = 1; size <= 10; ++size) append_byte(tables, (run << 4) | size);
  return tables;
}

class Scan {
public:
  explicit Scan(std::string& output) : output_(output) {}

  void block(const std::array<int, 64>& coefficients, int& previous_dc) {
    const int difference = coefficients[0] - previous_dc;
    previous_dc = coefficients[0];
    const unsigned dc_size = magnitude_bits(difference);
    put(dc_size, 4);
    amplitude(difference, dc_size);
    unsigned zeros = 0;
    for (size_t i = 1; i < zigzag.size(); ++i) {
      const int value = coefficients[zigzag[i]];
      if (!value) { ++zeros; continue; }
      while (zeros >= 16) {
        put(1, 8);
        zeros -= 16;
      }
      const unsigned size = magnitude_bits(value);
      put(2 + zeros * 10 + size - 1, 8);
      amplitude(value, size);
      zeros = 0;
    }
    if (zeros) put(0, 8);
  }

  void finish() {
    if (pending_) {
      const unsigned padding = 8 - pending_;
      put((1u << padding) - 1, padding);
    }
  }

private:
  static unsigned magnitude_bits(int value) {
    unsigned size = 0;
    for (unsigned magnitude = static_cast<unsigned>(std::abs(value)); magnitude; magnitude >>= 1) ++size;
    return size;
  }

  void amplitude(int value, unsigned size) {
    const int encoded = value < 0 ? value + (1 << size) - 1 : value;
    put(static_cast<unsigned>(encoded), size);
  }

  void put(unsigned value, unsigned size) {
    bits_ = (bits_ << size) | value;
    pending_ += size;
    while (pending_ >= 8) {
      pending_ -= 8;
      const auto byte = (bits_ >> pending_) & 255;
      append_byte(output_, byte);
      if (byte == 0xff) append_byte(output_, 0);
    }
    bits_ &= (1u << pending_) - 1;
  }

  std::string& output_;
  unsigned bits_ = 0, pending_ = 0;
};

std::array<int, 64> transformed(const Block& samples) {
  static const auto basis = [] {
    Block result{};
    const double pi = std::acos(-1.0);
    for (int frequency = 0; frequency < 8; ++frequency) {
      const double scale = frequency == 0 ? std::sqrt(0.125) : 0.5;
      for (int x = 0; x < 8; ++x)
        result[frequency * 8 + x] = scale * std::cos((2 * x + 1) * frequency * pi / 16);
    }
    return result;
  }();
  Block rows{};
  for (int y = 0; y < 8; ++y)
    for (int u = 0; u < 8; ++u)
      for (int x = 0; x < 8; ++x) rows[y * 8 + u] += samples[y * 8 + x] * basis[u * 8 + x];

  std::array<int, 64> coefficients{};
  for (int v = 0; v < 8; ++v) {
    for (int u = 0; u < 8; ++u) {
      double sum = 0;
      for (int y = 0; y < 8; ++y) sum += rows[y * 8 + u] * basis[v * 8 + y];
      const int i = v * 8 + u;
      coefficients[i] = static_cast<int>(std::lround(sum / quant[i]));
    }
  }
  return coefficients;
}

}  // namespace

std::string encode_jpeg(const Tensor& image) {
  const auto& shape = image.shape();
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0)
    throw std::invalid_argument("JPEG output requires one RGB image");
  const size_t height = shape[2], width = shape[3];
  if (width > 65535 || height > 65535) throw std::overflow_error("JPEG dimensions exceed 65535 pixels");
  const auto pixels = image.data();
  const auto sample = [&](size_t channel, size_t y, size_t x) {
    const float pixel = pixels[(channel * height + y) * width + x];
    if (!std::isfinite(pixel)) throw std::runtime_error("Diffusion produced a non-finite pixel");
    return 255.0 * std::clamp(pixel, 0.0f, 1.0f);
  };

  std::string jpeg;
  marker(jpeg, 0xd8);
  segment(jpeg, 0xe0, std::string_view("JFIF\0\1\1\0\0\1\0\1\0\0", 14));
  std::string quantization(1, '\0');
  for (int i : zigzag) append_byte(quantization, quant[i]);
  segment(jpeg, 0xdb, quantization);
  std::string frame(1, '\10');
  append_word(frame, static_cast<unsigned>(height));
  append_word(frame, static_cast<unsigned>(width));
  append_byte(frame, 3);
  for (unsigned component = 1; component <= 3; ++component) {
    append_byte(frame, component);
    append_byte(frame, 0x11);
    append_byte(frame, 0);
  }
  segment(jpeg, 0xc0, frame);
  segment(jpeg, 0xc4, huffman_tables());
  segment(jpeg, 0xda, std::string_view("\3\1\0\2\0\3\0\0\x3f\0", 10));

  Scan scan(jpeg);
  std::array<int, 3> previous_dc{};
  for (size_t y = 0; y < height; y += 8) {
    for (size_t x = 0; x < width; x += 8) {
      std::array<Block, 3> planes{};
      for (size_t dy = 0; dy < 8; ++dy) {
        for (size_t dx = 0; dx < 8; ++dx) {
          const size_t sx = std::min(x + dx, width - 1), sy = std::min(y + dy, height - 1);
          const double r = sample(0, sy, sx), g = sample(1, sy, sx), b = sample(2, sy, sx);
          const size_t i = dy * 8 + dx;
          planes[0][i] = 0.299 * r + 0.587 * g + 0.114 * b - 128;
          planes[1][i] = -0.168736 * r - 0.331264 * g + 0.5 * b;
          planes[2][i] = 0.5 * r - 0.418688 * g - 0.081312 * b;
        }
      }
      for (size_t channel = 0; channel < planes.size(); ++channel)
        scan.block(transformed(planes[channel]), previous_dc[channel]);
    }
  }
  scan.finish();
  marker(jpeg, 0xd9);
  return jpeg;
}

}  // namespace elman
