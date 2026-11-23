#include "IO/Png.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace elman {
namespace {

void append_u32(std::string& bytes, uint32_t value) {
  for (int shift = 24; shift >= 0; shift -= 8) bytes.push_back(static_cast<char>((value >> shift) & 255));
}

void append_chunk(std::string& png, std::string_view type, std::string_view data) {
  if (data.size() > 0x7fffffff) throw std::overflow_error("PNG chunk exceeds the size limit");
  append_u32(png, static_cast<uint32_t>(data.size()));
  const size_t start = png.size();
  png.append(type);
  png.append(data);
  uint32_t crc = 0xffffffff;
  for (size_t i = start; i < png.size(); ++i) {
    crc ^= static_cast<unsigned char>(png[i]);
    for (int bit = 0; bit < 8; ++bit) crc = (crc >> 1) ^ (crc & 1 ? 0xedb88320 : 0);
  }
  append_u32(png, ~crc);
}

std::string zlib_stored(std::string_view bytes) {
  std::string result = "\x78\x01";
  uint32_t first = 1, second = 0;
  for (size_t offset = 0; offset < bytes.size();) {
    const std::string_view block = bytes.substr(offset, 65535);
    offset += block.size();
    result.push_back(offset == bytes.size() ? '\1' : '\0');
    const uint16_t length = static_cast<uint16_t>(block.size());
    for (uint16_t word : {length, static_cast<uint16_t>(~length)}) {
      result.push_back(static_cast<char>(word & 255));
      result.push_back(static_cast<char>(word >> 8));
    }
    result.append(block);
    for (unsigned char byte : block) {
      first = (first + byte) % 65521;
      second = (second + first) % 65521;
    }
  }
  append_u32(result, (second << 16) | first);
  return result;
}

}  // namespace

std::string encode_png(const Tensor& image) {
  const Shape& shape = image.shape();
  if (shape.size() != 4 || shape[0] != 1 || shape[1] != 3 || shape[2] <= 0 || shape[3] <= 0)
    throw std::invalid_argument("PNG output requires one RGB image");
  const size_t height = shape[2], width = shape[3];
  const uint64_t bytes = (3 * uint64_t(width) + 1) * height;
  if (bytes + (bytes / 65535 + 1) * 5 + 6 > 0x7fffffff)
    throw std::overflow_error("PNG image exceeds the encoder's chunk size limit");
  const std::vector<float> pixels = image.data();
  std::string scanlines;
  scanlines.reserve(static_cast<size_t>(bytes));
  for (size_t y = 0; y < height; ++y) {
    scanlines.push_back('\0');  // Filter type None.
    for (size_t x = 0; x < width; ++x) {
      for (size_t channel = 0; channel < 3; ++channel) {
        const float pixel = pixels[(channel * height + y) * width + x];
        if (!std::isfinite(pixel)) throw std::runtime_error("Diffusion produced a non-finite pixel");
        scanlines.push_back(static_cast<char>(std::lround(std::clamp(pixel, 0.0f, 1.0f) * 255.0f)));
      }
    }
  }
  std::string png = "\x89PNG\r\n\x1a\n";
  std::string header;
  append_u32(header, static_cast<uint32_t>(width));
  append_u32(header, static_cast<uint32_t>(height));
  header.append("\x08\x02\0\0\0", 5);  // RGB, 8 bits per channel, no interlace.
  append_chunk(png, "IHDR", header);
  append_chunk(png, "IDAT", zlib_stored(scanlines));
  append_chunk(png, "IEND", {});
  return png;
}

}  // namespace elman
