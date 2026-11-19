#include "Inference/GraphSession.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "Core/AllocationPool.h"
#include "Core/Device.h"
#include "Core/TensorDispatch.h"
#include "IO/Json.h"
#include "IO/TensorArchive.h"
#include "Operators/inference_ops.h"

namespace elman {
namespace {

using Operands = std::vector<Tensor>;
using Operation = Tensor (*)(const Operands&, const Json&);

std::vector<int> integers(const Json& object, const char* key, std::vector<int> defaults = {}) {
  if (!object.contains(key)) return defaults;
  std::vector<int> result;
  for (const auto& item : object.at(key).array()) result.push_back(item.integer());
  return result;
}

int axis(int dim, int rank) {
  if (dim < 0) dim += rank;
  if (dim < 0 || dim >= rank) throw std::invalid_argument("Graph axis out of range");
  return dim;
}

std::vector<int64_t> indices(const Tensor& tensor) {
  if (tensor.dtype() == DType::Int64) return tensor.int64_data();
  if (tensor.dtype() != DType::Int32) throw std::invalid_argument("Graph index tensors require Int32 or Int64");
  auto values = tensor.int_data();
  return std::vector<int64_t>(values.begin(), values.end());
}

Tensor graph_reshape(const Operands& x, const Json& p) {
  const auto requested = integers(p, "shape");
  const auto dims = x.size() == 2 ? indices(x[1]) : std::vector<int64_t>(requested.begin(), requested.end());
  const bool allow_zero = p.int_or("allowzero", 0) != 0;
  Shape shape;
  for (size_t i = 0; i < dims.size(); ++i) {
    if (dims[i] < -1 || dims[i] > std::numeric_limits<int>::max()) throw std::invalid_argument("Graph reshape dimension out of range");
    shape.push_back(!allow_zero && dims[i] == 0 ? x[0].shape().at(i) : static_cast<int>(dims[i]));
  }
  return reshape(x[0], std::move(shape));
}

Tensor flatten(const Operands& x, const Json& p) {
  const auto& shape = x[0].shape();
  const int rank = static_cast<int>(shape.size());
  int dim = p.int_or("axis", 1);
  if (dim < 0) dim += rank;
  if (dim < 0 || dim > rank) throw std::invalid_argument("Graph flatten axis out of range");
  return reshape(x[0], {static_cast<int>(shape_element_count(Shape(shape.begin(), shape.begin() + dim))),
                       static_cast<int>(shape_element_count(Shape(shape.begin() + dim, shape.end())))});
}

Tensor graph_transpose(const Operands& x, const Json& p) {
  if (x.size() == 2) {
    std::vector<int> axes;
    for (const auto value : indices(x[1])) {
      if (value < 0 || value >= static_cast<int64_t>(x[0].shape().size()))
        throw std::invalid_argument("Graph permutation axis out of range");
      axes.push_back(int(value));
    }
    return permute(x[0], axes);
  }
  if (p.contains("perm")) return permute(x[0], integers(p, "perm"));
  std::vector<int> axes;
  for (int i = static_cast<int>(x[0].shape().size()) - 1; i >= 0; --i) axes.push_back(i);
  return permute(x[0], axes);
}

Tensor graph_gather(const Operands& x, const Json& p) {
  const auto& shape = x[0].shape();
  const int dim = axis(p.int_or("axis", 0), static_cast<int>(shape.size()));
  auto values = indices(x[1]);
  for (auto& index : values) {
    if (index < 0) index += shape[dim];
    if (index < 0 || index >= shape[dim]) throw std::out_of_range("Graph gather index out of range");
  }
  Shape result(shape.begin(), shape.begin() + dim);
  result.insert(result.end(), x[1].shape().begin(), x[1].shape().end());
  result.insert(result.end(), shape.begin() + dim + 1, shape.end());
  return reshape(index_select(x[0], dim, Tensor::from_int64_data({static_cast<int>(values.size())}, values, x[0].backend())), result);
}

Tensor gemm(const Operands& x, const Json& p) {
  auto left = p.int_or("transA", 0) ? transpose(x[0], 0, 1) : x[0];
  auto right = p.int_or("transB", 0) ? transpose(x[1], 0, 1) : x[1];
  auto product = matmul(left, right);
  const double alpha = p.number_or("alpha", 1), beta = p.number_or("beta", 1);
  if (alpha != 1) product = mul(product, alpha);
  return x.size() == 3 && x[2].defined() && beta != 0 ? add(product, beta == 1 ? x[2] : mul(x[2], beta)) : product;
}

std::pair<int, int> spatial_geometry(const Json& p, int default_stride = 1) {
  const auto strides = integers(p, "strides", {default_stride, default_stride});
  const auto pads = integers(p, "pads", {0, 0, 0, 0});
  if (strides.size() != 2 || strides[0] <= 0 || strides[0] != strides[1] || pads.size() != 4 ||
      pads[0] < 0 || !std::all_of(pads.begin(), pads.end(), [&](int pad) { return pad == pads[0]; }) ||
      p.string_or("auto_pad", "NOTSET") != "NOTSET" || integers(p, "dilations", {1, 1}) != std::vector<int>{1, 1})
    throw std::invalid_argument("Graph spatial operator requires square strides, symmetric explicit padding and unit dilation");
  return {strides[0], pads[0]};
}

Tensor convolution(const Operands& x, const Json& p) {
  if (p.int_or("group", 1) != 1) throw std::invalid_argument("Graph convolution requires one group");
  const auto [stride, pad] = spatial_geometry(p);
  return conv2d(x[0], x[1], x.size() == 3 ? x[2] : Tensor{}, stride, pad);
}

Tensor pool(const Operands& x, const Json& p, bool maximum) {
  const auto kernel = integers(p, "kernel_shape");
  const auto [stride, pad] = spatial_geometry(p);
  if (kernel.size() != 2 || kernel[0] != kernel[1] || kernel[0] < 1 || p.int_or("ceil_mode", 0) ||
      (!maximum && pad && !p.int_or("count_include_pad", 0)))
    throw std::invalid_argument("Unsupported graph pooling geometry");
  return maximum ? max_pool2d(x[0], kernel[0], stride, pad) : avg_pool2d(x[0], kernel[0], stride, pad);
}

Tensor batch_normalization(const Operands& x, const Json& p) {
  if (p.int_or("training_mode", 0) || x[0].shape().size() < 2)
    throw std::invalid_argument("Graph batch normalization requires inference mode and a channel axis");
  Shape affine(x[0].shape().size(), 1);
  affine[1] = x[0].shape()[1];
  auto scale = mul(x[1], rsqrt(add(x[4], p.number_or("epsilon", 1e-5))));
  return add(mul(sub(x[0], reshape(x[3], affine)), reshape(scale, affine)), reshape(x[2], affine));
}

Tensor scalar(const Tensor& like, double value) {
  return mul(Tensor::ones({}, like.backend(), false, like.dtype()), value);
}

Tensor fill_spatial(const Tensor& value, int dim, int before, int after, double fill) {
  if (!before && !after) return value;
  auto shape = value.shape();
  auto edge = [&](int count) {
    shape[dim] = count;
    if (fill == 0) return Tensor::zeros(shape, value.backend(), false, value.dtype());
    return expand(scalar(value, fill), shape);
  };
  std::vector<Tensor> pieces;
  if (before) pieces.push_back(edge(before));
  pieces.push_back(value);
  if (after) pieces.push_back(edge(after));
  return concat(pieces, dim);
}

struct NhwcGeometry {
  int stride_h, stride_w, top, bottom, left, right;

  NhwcGeometry(const Tensor& chw, int h, int w, const Json& attributes)
      : stride_h(attributes.at("stride_h").integer()), stride_w(attributes.at("stride_w").integer()),
        top(0), bottom(0), left(0), right(0) {
    if (stride_h <= 0 || stride_w <= 0 || h <= 0 || w <= 0 || chw.shape()[2] <= 0 || chw.shape()[3] <= 0)
      throw std::invalid_argument("Invalid NHWC spatial dimensions");
    if (attributes.bool_or("same", false)) {
      const auto padding = [](int length, int kernel, int stride) {
        const int64_t total = std::max<int64_t>(0, int64_t((length - 1) / stride) * stride + kernel - length);
        if (int64_t(length) + total > std::numeric_limits<int>::max())
          throw std::invalid_argument("NHWC padding exceeds the supported extent");
        return int(total);
      };
      const auto height = padding(chw.shape()[2], h, stride_h), width = padding(chw.shape()[3], w, stride_w);
      top = height / 2;
      bottom = height - top;
      left = width / 2;
      right = width - left;
    }
  }

  Tensor pad(const Tensor& chw, double fill = 0) const {
    return fill_spatial(fill_spatial(chw, 2, top, bottom, fill), 3, left, right, fill);
  }
  int kernel_stride() const { return stride_h == stride_w ? stride_h : 1; }
  Tensor output(const Tensor& nchw) const {
    if (stride_h == stride_w) return permute(nchw, {0, 2, 3, 1});
    const auto rows = slice(nchw, 2, 0, nchw.shape()[2], stride_h);
    return permute(slice(rows, 3, 0, rows.shape()[3], stride_w), {0, 2, 3, 1});
  }
};

Tensor nhwc_convolution(const Operands& x, const Json& p) {
  if (x[0].shape().size() != 4 || x[1].shape().size() != 4)
    throw std::invalid_argument("NHWC convolution requires rank-four tensors");
  const auto nchw = permute(x[0], {0, 3, 1, 2});
  const auto& filter = x[1].shape();
  const NhwcGeometry geometry(nchw, filter[1], filter[2], p);
  const auto input = geometry.pad(nchw);
  const auto bias = x.size() == 3 ? x[2] : Tensor{};
  if (!p.bool_or("depthwise", false)) {
    const auto weights = permute(x[1], {0, 3, 1, 2});
    const int groups = p.int_or("group", 1);
    if (groups == 1) return geometry.output(conv2d(input, weights, bias, geometry.kernel_stride()));
    const auto channels = input.shape()[1], outputs = filter[0];
    if (groups <= 0 || channels % groups || outputs % groups || channels / groups != filter[3] ||
        (bias.defined() && bias.shape() != Shape{outputs}))
      throw std::invalid_argument("Invalid NHWC grouped convolution filter");
    std::vector<Tensor> result;
    const int c = channels / groups, o = outputs / groups;
    for (int group = 0; group < groups; ++group)
      result.push_back(conv2d(slice(input, 1, c * group, c * (group + 1)),
                             slice(weights, 0, o * group, o * (group + 1)),
                             bias.defined() ? slice(bias, 0, o * group, o * (group + 1)) : Tensor{},
                             geometry.kernel_stride()));
    return geometry.output(concat(result, 1));
  }
  const int channels = nchw.shape()[1];
  if (filter[0] != 1 || channels <= 0 || filter[3] <= 0 || filter[3] % channels ||
      (bias.defined() && bias.shape() != Shape{filter[3]}))
    throw std::invalid_argument("Invalid NHWC depthwise convolution filter");
  const int multiplier = filter[3] / channels;
  const auto weights = permute(x[1], {3, 0, 1, 2});
  std::vector<Tensor> result;
  for (int channel = 0; channel < channels; ++channel) {
    const int start = channel * multiplier, end = start + multiplier;
    result.push_back(conv2d(slice(input, 1, channel, channel + 1), slice(weights, 0, start, end),
                           bias.defined() ? slice(bias, 0, start, end) : Tensor{}, geometry.kernel_stride()));
  }
  return geometry.output(concat(result, 1));
}

Tensor nhwc_pool(const Operands& x, const Json& p) {
  if (x[0].shape().size() != 4) throw std::invalid_argument("NHWC pooling requires a rank-four input");
  const auto input = permute(x[0], {0, 3, 1, 2});
  const int kernel = p.at("kernel").integer();
  const NhwcGeometry geometry(input, kernel, kernel, p);
  const int stride = geometry.kernel_stride();
  if (p.at("maximum").boolean())
    return geometry.output(max_pool2d(geometry.pad(input, -std::numeric_limits<double>::infinity()), kernel, stride));
  auto result = avg_pool2d(geometry.pad(input), kernel, stride);
  if (geometry.top || geometry.bottom || geometry.left || geometry.right) {
    const auto& s = input.shape();
    const auto ones = Tensor::ones({1, 1, s[2], s[3]}, input.backend(), false, input.dtype());
    result = div(result, avg_pool2d(geometry.pad(ones), kernel, stride));
  }
  return geometry.output(result);
}

Tensor fully_connected(const Operands& x, const Json& p) {
  const auto& weight = x[1].shape();
  if (weight.size() != 2 || weight[1] <= 0 || x[0].numel() % weight[1])
    throw std::invalid_argument("Invalid fully connected input size");
  auto input = x[0];
  if (!p.bool_or("keep_dims", false)) input = reshape(input, {-1, weight[1]});
  return linear(input, x[1], x.size() == 3 ? x[2] : Tensor{});
}

template<Tensor (*Function)(const Tensor&, int, bool)>
Tensor graph_reduce(const Operands& x, const Json& p) {
  auto input = x[0];
  std::set<int, std::greater<int>> dimensions;
  for (auto dim : indices(x[1])) {
    if (dim < -int64_t(input.shape().size()) || dim >= int64_t(input.shape().size()))
      throw std::invalid_argument("Reduction axis out of range");
    dimensions.insert(axis(int(dim), int(input.shape().size())));
  }
  for (auto dim : dimensions) input = Function(input, dim, p.bool_or("keep_dims", false));
  if constexpr (Function == sum) if (input.dtype() != x[0].dtype()) input = cast(input, x[0].dtype());
  return input;
}

Tensor graph_matmul(const Operands& x, const Json& p) {
  auto operand = [&](size_t index, const char* attribute) {
    auto value = x[index];
    if (!p.bool_or(attribute, false)) return value;
    value = transpose(value, -2, -1);
    return p.bool_or("adjoint", false) && is_complex(value.dtype()) ? conj(value) : value;
  };
  return matmul(operand(0, "transA"), operand(1, "transB"));
}

Tensor true_divide(const Tensor& a, const Tensor& b) {
  if (is_standard_integer(a.dtype()) && is_standard_integer(b.dtype()))
    return div(cast(a, DType::Float32), cast(b, DType::Float32));
  return div(a, b);
}

Tensor scalar_binary(const Operands& x, const Json& p) {
  const auto backend = x[0].backend();
  Tensor scalar_value;
  if (p.contains("integer")) {
    const auto value = std::stoll(p.at("integer").string());
    scalar_value = p.bool_or("boolean", false) ? Tensor::from_raw({}, DType::Bool, {uint8_t(value != 0)}, backend) :
                                                 Tensor::from_int64_data({}, {value}, backend);
  } else {
    const auto value = p.at("value").number();
    scalar_value = is_standard_integer(x[0].dtype()) ? Tensor::from_data({}, {float(value)}, backend) : scalar(x[0], value);
  }
  const auto& left = p.bool_or("reverse", false) ? scalar_value : x[0];
  const auto& right = p.bool_or("reverse", false) ? x[0] : scalar_value;
  const auto op = p.at("op").string();
  if (op == "Add") return add(left, right);
  if (op == "Sub") return sub(left, right);
  if (op == "Mul") return mul(left, right);
  if (op == "Div") return div(left, right);
  if (op == "TrueDivide") return true_divide(left, right);
  if (op == "Eq") return eq(left, right);
  if (op == "Ne") return ne(left, right);
  if (op == "Gt") return gt(left, right);
  if (op == "Ge") return ge(left, right);
  if (op == "Lt") return lt(left, right);
  if (op == "Le") return le(left, right);
  throw std::invalid_argument("Unknown graph scalar operation " + op);
}

Tensor graph_squeeze(const Operands& x, const Json& p) {
  auto value = x[0];
  const auto axes = integers(p, "axes");
  if (axes.empty()) return squeeze(value);
  std::set<int, std::greater<int>> dimensions;
  for (int dim : axes) dimensions.insert(axis(dim, int(value.shape().size())));
  for (int dim : dimensions) {
    if (value.shape()[dim] != 1 && p.bool_or("strict", true))
      throw std::invalid_argument("Graph squeeze requires a singleton dimension");
    value = squeeze(value, dim);
  }
  return value;
}

Tensor graph_padding(const Operands& x, const Json& p) {
  auto value = x[0];
  const auto pads = x.size() > 1 ? indices(x[1]) : [&] {
    const auto v = integers(p, "pads");
    return std::vector<int64_t>(v.begin(), v.end());
  }();
  if (pads.size() != value.shape().size() * 2) throw std::invalid_argument("Graph padding rank mismatch");
  for (size_t dim = 0; dim < value.shape().size(); ++dim) {
    const auto left = pads[2 * dim], right = pads[2 * dim + 1];
    if (left < 0 || right < 0 || left > INT32_MAX || right > INT32_MAX ||
        left + right + value.shape()[dim] > INT32_MAX)
      throw std::invalid_argument("Invalid graph padding extent");
    value = fill_spatial(value, int(dim), int(left), int(right), p.number_or("fill", 0));
  }
  return value;
}

Tensor adaptive_average_pool(const Operands& x, const Json& p) {
  const auto& shape = x[0].shape();
  const int height = p.at("height").integer(), width = p.at("width").integer();
  if (shape.size() < 3 || shape.size() > 4 || height <= 0 || width <= 0 ||
      shape[shape.size() - 2] <= 0 || shape.back() <= 0)
    throw std::invalid_argument("Invalid graph adaptive pooling geometry");
  const auto h = shape[shape.size() - 2], w = shape.back();
  if (h % height == 0 && w % width == 0) {
    auto grouped = shape;
    grouped.pop_back();
    grouped.back() = height;
    grouped.insert(grouped.end(), {h / height, width, w / width});
    return mean(mean(reshape(x[0], grouped), -1), -2);
  }
  std::vector<Tensor> rows;
  for (int i = 0; i < height; ++i) {
    auto section = slice(x[0], -2, int(int64_t(i) * h / height), int((int64_t(i + 1) * h + height - 1) / height));
    std::vector<Tensor> cells;
    for (int j = 0; j < width; ++j) {
      auto tile = slice(section, -1, int(int64_t(j) * w / width), int((int64_t(j + 1) * w + width - 1) / width));
      cells.push_back(mean(mean(tile, -1, true), -2, true));
    }
    rows.push_back(concat(cells, -1));
  }
  return concat(rows, -2);
}

Tensor graph_strided_slice(const Operands& x, const Json& p) {
  auto value = x[0];
  const auto begin = indices(x[1]), end = indices(x[2]), stride = indices(x[3]);
  if (begin.size() != end.size() || begin.size() != stride.size() || begin.size() > value.shape().size() ||
      p.int_or("new_axis_mask", 0) || p.int_or("ellipsis_mask", 0))
    throw std::invalid_argument("Unsupported graph strided slice index geometry");
  int dim = 0;
  for (size_t index = 0; index < begin.size(); ++index) {
    const int64_t length = value.shape()[dim], step = stride[index];
    const int bit = 1 << index;
    if (!step || step > INT32_MAX || step < -INT32_MAX)
      throw std::invalid_argument("Invalid graph strided slice stride");
    const auto normalize = [length](int64_t item) { return item < 0 ? length + item : item; };
    auto start = p.int_or("begin_mask", 0) & bit ? (step > 0 ? 0 : length - 1) : normalize(begin[index]);
    auto stop = p.int_or("end_mask", 0) & bit ? (step > 0 ? length : -1) : normalize(end[index]);
    if (p.int_or("shrink_axis_mask", 0) & bit) {
      if (start < 0 || start >= length) throw std::invalid_argument("Graph strided slice index out of bounds");
      value = squeeze(slice(value, dim, int(start), int(start + 1)), dim);
      continue;
    }
    if (step > 0) {
      start = std::clamp<int64_t>(start, 0, length);
      stop = std::clamp<int64_t>(stop, 0, length);
      value = slice(value, dim, int(start), int(stop), int(std::min(step, std::max<int64_t>(1, length))));
    } else {
      std::vector<int64_t> reversed;
      start = std::clamp<int64_t>(start, -1, length - 1);
      stop = std::clamp<int64_t>(stop, -1, length - 1);
      for (int64_t i = start; i > stop; i += step) reversed.push_back(i);
      value = index_select(value, dim, Tensor::from_int64_data({int(reversed.size())}, reversed, value.backend()));
    }
    ++dim;
  }
  return value;
}

template<Tensor (*Function)(const Tensor&, const Tensor&)>
Tensor binary(const Operands& x, const Json&) { return Function(x[0], x[1]); }

template<Tensor (*Function)(const Tensor&)>
Tensor unary(const Operands& x, const Json&) { return Function(x[0]); }

struct OperationInfo { Operation run; size_t minimum, maximum; };

const std::unordered_map<std::string, OperationInfo>& supported_operations() {
  static const std::unordered_map<std::string, OperationInfo> operations{
      {"Identity", {[](const Operands& x, const Json&) { return x[0]; }, 1, 1}},
      {"Relu", {unary<relu>, 1, 1}}, {"Exp", {unary<exp>, 1, 1}}, {"Sqrt", {unary<sqrt>, 1, 1}},
      {"Tanh", {unary<tanh>, 1, 1}}, {"Abs", {unary<abs>, 1, 1}},
      {"Silu", {unary<silu>, 1, 1}}, {"Gelu", {unary<gelu>, 1, 1}}, {"ExactGelu", {unary<exact_gelu>, 1, 1}},
      {"Log", {unary<log>, 1, 1}}, {"Rsqrt", {unary<rsqrt>, 1, 1}},
      {"Minimum", {binary<minimum>, 2, 2}}, {"Maximum", {binary<maximum>, 2, 2}},
      {"Square", {[](const Operands& x, const Json&) { return mul(x[0], x[0]); }, 1, 1}},
      {"Neg", {[](const Operands& x, const Json&) { return mul(x[0], -1); }, 1, 1}},
      {"LeakyRelu", {[](const Operands& x, const Json& p) { return leaky_relu(x[0], p.at("alpha").number()); }, 1, 1}},
      {"Clip", {[](const Operands& x, const Json& p) {
          return minimum(maximum(x[0], scalar(x[0], p.at("min").number())), scalar(x[0], p.at("max").number()));
        }, 1, 1}},
      {"Add", {binary<add>, 2, 2}}, {"Sub", {binary<sub>, 2, 2}}, {"Mul", {binary<mul>, 2, 2}},
      {"Div", {binary<div>, 2, 2}}, {"TrueDivide", {binary<true_divide>, 2, 2}}, {"MatMul", {graph_matmul, 2, 2}},
      {"Eq", {binary<eq>, 2, 2}}, {"Ne", {binary<ne>, 2, 2}},
      {"Gt", {binary<gt>, 2, 2}}, {"Ge", {binary<ge>, 2, 2}},
      {"Lt", {binary<lt>, 2, 2}}, {"Le", {binary<le>, 2, 2}},
      {"ScalarBinary", {scalar_binary, 1, 1}},
      {"Sigmoid", {[](const Operands& x, const Json&) {
          auto one = Tensor::ones({}, x[0].backend(), false, x[0].dtype());
          return div(one, add(exp(mul(x[0], -1.0)), 1.0));
        }, 1, 1}},
      {"Gemm", {gemm, 2, 3}}, {"Reshape", {graph_reshape, 1, 2}}, {"Flatten", {flatten, 1, 1}},
      {"FlattenRange", {[](const Operands& x, const Json& p) {
          const auto& previous = x[0].shape();
          if (previous.empty()) return reshape(x[0], {1});
          const int start = axis(p.int_or("start", 0), int(previous.size()));
          const int end = axis(p.int_or("end", -1), int(previous.size()));
          if (end < start) throw std::invalid_argument("Invalid graph flatten axis range");
          const auto width = shape_element_count(Shape(previous.begin() + start, previous.begin() + end + 1));
          if (width > INT32_MAX) throw std::invalid_argument("Graph flatten extent exceeds the tensor limit");
          Shape shape(previous.begin(), previous.begin() + start);
          shape.push_back(int(width));
          shape.insert(shape.end(), previous.begin() + end + 1, previous.end());
          return reshape(x[0], std::move(shape));
        }, 1, 1}},
      {"SwapAxes", {[](const Operands& x, const Json& p) {
          return transpose(x[0], p.at("a").integer(), p.at("b").integer());
        }, 1, 1}},
      {"Size", {[](const Operands& x, const Json& p) {
          const auto& shape = x[0].shape();
          return Tensor::from_int64_data({}, {shape.at(axis(p.at("axis").integer(), int(shape.size())))}, x[0].backend());
        }, 1, 1}},
      {"Rank", {[](const Operands& x, const Json&) {
          return Tensor::from_int64_data({}, {int64_t(x[0].shape().size())}, x[0].backend());
        }, 1, 1}},
      {"AssertFalse", {[](const Operands& x, const Json&) {
          if (x[0].numel() != 1 || x[0].dtype() != DType::Bool || x[0].raw_data().at(0))
            throw std::invalid_argument("Graph runtime assertion failed");
          return x[0];
        }, 1, 1}},
      {"Transpose", {graph_transpose, 1, 2}}, {"Gather", {graph_gather, 2, 2}},
      {"Softmax", {[](const Operands& x, const Json& p) {
          const auto beta = p.number_or("beta", 1);
          return softmax(beta == 1 ? x[0] : mul(x[0], beta), p.int_or("axis", -1));
        }, 1, 1}},
      {"LogSoftmax", {[](const Operands& x, const Json& p) { return log_softmax(x[0], p.int_or("axis", -1)); }, 1, 1}},
      {"Concat", {[](const Operands& x, const Json& p) { return concat(x, p.at("axis").integer()); }, 1, 1024}},
      {"Mean", {graph_reduce<mean>, 2, 2}}, {"Sum", {graph_reduce<sum>, 2, 2}},
      {"ReduceMax", {graph_reduce<amax>, 2, 2}},
      {"ExpandDims", {[](const Operands& x, const Json& p) {
          if (x.size() == 1) return unsqueeze(x[0], p.at("axis").integer());
          const auto dims = indices(x[1]);
          if (dims.size() != 1 || dims[0] < INT32_MIN || dims[0] > INT32_MAX)
            throw std::invalid_argument("Graph expand_dims requires one integer axis");
          return unsqueeze(x[0], int(dims[0]));
        }, 1, 2}},
      {"Squeeze", {graph_squeeze, 1, 1}},
      {"Stack", {[](const Operands& x, const Json& p) {
          const auto dim = p.int_or("axis", 0);
          std::vector<Tensor> expanded;
          for (const auto& value : x) expanded.push_back(unsqueeze(value, dim));
          return concat(expanded, dim);
        }, 1, 1024}},
      {"StridedSlice", {graph_strided_slice, 4, 4}},
      {"Pad", {graph_padding, 1, 2}},
      {"AdaptiveAvgPool2d", {adaptive_average_pool, 1, 1}},
      {"ChannelsLast", {[](const Operands& x, const Json&) {
          const auto rank = int(x[0].shape().size());
          if (rank < 3) return x[0];
          std::vector<int> order{0};
          for (int dim = 2; dim < rank; ++dim) order.push_back(dim);
          order.push_back(1);
          return permute(x[0], order);
        }, 1, 1}},
      {"BatchReshape", {[](const Operands& x, const Json& p) {
          auto shape = integers(p, "shape");
          shape.insert(shape.begin(), x[0].shape().at(0));
          return reshape(x[0], shape);
        }, 1, 1}},
      {"Cast", {[](const Operands& x, const Json& p) { return cast(x[0], parse_dtype(p.at("to").string())); }, 1, 1}},
      {"Shape", {[](const Operands& x, const Json& p) {
          const auto& shape = x[0].shape();
          const int rank = static_cast<int>(shape.size());
          auto normalize = [rank](int dim) { return std::clamp(dim < 0 ? dim + rank : dim, 0, rank); };
          int start = normalize(p.int_or("start", 0)), end = std::max(start, normalize(p.int_or("end", rank)));
          auto result = Tensor::from_int64_data({end - start}, std::vector<int64_t>(shape.begin() + start, shape.begin() + end), x[0].backend());
          return p.contains("to") ? cast(result, parse_dtype(p.at("to").string())) : result;
        }, 1, 1}},
      {"ChannelBias", {[](const Operands& x, const Json&) {
          if (x[0].shape().size() < 2 || x[1].shape() != Shape{x[0].shape()[1]})
            throw std::invalid_argument("Graph channel bias requires a matching channel axis");
          Shape shape(x[0].shape().size(), 1);
          shape[1] = x[1].shape()[0];
          return add(x[0], reshape(x[1], shape));
        }, 2, 2}},
      {"DepthwiseFilter", {[](const Operands& x, const Json&) {
          const auto& shape = x[0].shape();
          if (shape.size() != 4 || int64_t(shape[2]) * shape[3] > INT32_MAX)
            throw std::invalid_argument("Invalid graph depthwise filter");
          return reshape(x[0], {1, shape[0], shape[1], shape[2] * shape[3]});
        }, 1, 1}},
      {"Conv", {convolution, 2, 3}},
      {"NhwcConv", {nhwc_convolution, 2, 3}},
      {"NhwcPool", {nhwc_pool, 1, 1}},
      {"FullyConnected", {fully_connected, 2, 3}},
      {"MaxPool", {[](const Operands& x, const Json& p) { return pool(x, p, true); }, 1, 1}},
      {"AveragePool", {[](const Operands& x, const Json& p) { return pool(x, p, false); }, 1, 1}},
      {"GlobalAveragePool", {[](const Operands& x, const Json&) {
          auto value = x[0];
          for (int dim = int(value.shape().size()) - 1; dim >= 2; --dim) value = mean(value, dim, true);
          return value;
        }, 1, 1}},
      {"BatchNormalization", {batch_normalization, 5, 5}},
      {"AffineBatchNormalization", {[](const Operands& x, const Json& p) {
          const int dim = axis(p.int_or("axis", -1), int(x[0].shape().size()));
          const Shape channel{x[0].shape()[dim]};
          Shape broadcast(x[0].shape().size(), 1);
          broadcast[dim] = channel[0];
          auto weights = [&](size_t index) {
            if (x[index].shape() != channel) throw std::invalid_argument("Incompatible graph batch normalization weight");
            return reshape(x[index], broadcast);
          };
          auto scale = rsqrt(add(weights(2), p.number_or("epsilon", 1e-3)));
          if (x.size() > 3 && x[3].defined()) scale = mul(scale, weights(3));
          auto result = mul(sub(x[0], weights(1)), scale);
          return x.size() > 4 && x[4].defined() ? add(result, weights(4)) : result;
        }, 3, 5}},
      {"LayerNormalization", {[](const Operands& x, const Json& p) {
          if (x[0].shape().empty()) throw std::invalid_argument("Layer normalization requires a feature axis");
          const Shape features{x[0].shape().back()};
          const auto scale = x.size() > 1 && x[1].defined() ? x[1] :
            Tensor::ones(features, x[0].backend(), false, x[0].dtype());
          const auto bias = x.size() > 2 && x[2].defined() ? x[2] :
            Tensor::zeros(features, x[0].backend(), false, x[0].dtype());
          return layer_norm(x[0], scale, bias, p.number_or("epsilon", 1e-3));
        }, 1, 3}},
  };
  return operations;
}

TensorSignature signature(const Json& json) {
  TensorSignature result;
  for (const auto& [name, desc] : json.object()) {
    const bool has_type = desc.contains("dtype");
    result.emplace(name, TensorSpec{has_type ? parse_dtype(desc.at("dtype").string()) : DType::Float32,
                                   integers(desc, "shape"), !desc.contains("shape"), !has_type});
  }
  return result;
}

struct GraphNode {
  std::string name;
  Operation run;
  std::vector<size_t> reads;
  size_t output;
  Json attributes;
};

struct Graph {
  static constexpr size_t omitted = std::numeric_limits<size_t>::max();
  std::vector<GraphNode> nodes;
  std::unordered_map<std::string, size_t> ids;
  std::vector<size_t> readers;
  std::map<std::string, size_t> inputs, outputs;
  std::vector<std::pair<size_t, Tensor>> weights;
  size_t register_value(const std::string& name) {
    const auto id = ids.size();
    if (name.empty() || !ids.emplace(name, id).second) throw std::invalid_argument("Duplicate or empty graph output " + name);
    return id;
  }
  NamedTensors operator()(const NamedTensors& arguments) const {
    AllocationPool allocations(arguments.begin()->second.backend());
    std::vector<Tensor> values(ids.size());
    auto remaining = readers;
    for (const auto& [name, id] : inputs) values[id] = arguments.at(name);
    for (const auto& [id, tensor] : weights) values[id] = tensor;
    for (const auto& node : nodes) {
      Operands operands;
      operands.reserve(node.reads.size());
      for (size_t id : node.reads) operands.push_back(id == omitted ? Tensor{} : values[id]);
      try {
        values[node.output] = node.run(operands, node.attributes);
      } catch (const std::exception& error) {
        throw std::runtime_error("Graph node " + node.name + ": " + error.what());
      }
      for (size_t id : node.reads) if (id != omitted && --remaining[id] == 0) values[id] = {};
      if (!remaining[node.output]) values[node.output] = {};
    }
    NamedTensors result;
    for (const auto& [name, id] : outputs) result.emplace(name, values[id]);
    return result;
  }
};

}  // namespace

InferenceSession load_graph_session(const std::filesystem::path& path, Backend backend, int device) {
  DeviceScope owner(backend, device);
  const auto definition = Json::read(path);
  if (definition.at("format").string() != "elman.graph" || definition.at("version").integer() != 1)
    throw std::invalid_argument("Unsupported Elman graph format");
  const auto weights_file = std::filesystem::u8path(definition.string_or("initializers", ""));
  NamedTensors weights;
  if (!weights_file.empty()) {
    if (weights_file.has_root_path() || weights_file.filename() != weights_file || weights_file.extension() != ".safetensors")
      throw std::invalid_argument("Graph initializers must name a Safetensors file beside the graph");
    weights = load_tensor_archive(path.parent_path() / weights_file, backend);
  }
  return make_graph_session(definition, std::move(weights), backend, device);
}

InferenceSession make_graph_session(const Json& definition, NamedTensors weights, Backend backend, int device) {
  auto inputs = signature(definition.at("inputs"));
  auto outputs = signature(definition.at("outputs"));
  auto graph = std::make_shared<Graph>();
  for (const auto& input : inputs) {
    const auto value = definition.at("inputs").at(input.first).string_or("value", input.first);
    graph->inputs.emplace(input.first, graph->register_value(value));
  }
  for (auto& [name, tensor] : weights) graph->weights.emplace_back(graph->register_value(name), std::move(tensor));
  for (const auto& value : definition.at("nodes").array()) {
    const auto op = value.at("op").string();
    const auto info = supported_operations().find(op);
    if (info == supported_operations().end()) throw std::invalid_argument("Unsupported graph operation " + op);
    const auto output = value.at("output").string();
    auto attributes = value.contains("attributes") ? value.at("attributes") : Json(Json::Object{});
    attributes.object();
    GraphNode node{value.string_or("name", output), info->second.run, {}, 0, std::move(attributes)};
    const auto& read = value.at("inputs").array();
    if (read.size() < info->second.minimum || read.size() > info->second.maximum)
      throw std::invalid_argument("Wrong number of inputs for graph operation " + op);
    for (size_t i = 0; i < read.size(); ++i) {
      const auto operand = read[i].string();
      if (operand.empty() && i >= info->second.minimum) node.reads.push_back(Graph::omitted);
      else node.reads.push_back(graph->ids.at(operand));
    }
    node.output = graph->register_value(output);
    graph->nodes.push_back(std::move(node));
  }
  graph->readers.resize(graph->ids.size());
  for (const auto& node : graph->nodes)
    for (size_t id : node.reads) if (id != Graph::omitted) ++graph->readers[id];
  for (const auto& output : outputs) {
    const auto value = definition.at("outputs").at(output.first).string_or("value", output.first);
    const size_t id = graph->ids.at(value);
    graph->outputs.emplace(output.first, id);
    ++graph->readers[id];
  }
  return InferenceSession(backend, device, std::move(inputs), std::move(outputs),
      [graph = std::move(graph)](const NamedTensors& values) { return (*graph)(values); });
}

}  // namespace elman
