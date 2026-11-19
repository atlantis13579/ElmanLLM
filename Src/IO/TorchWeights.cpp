#include "IO/TorchWeights.h"
#include "IO/TorchScriptArchive.h"

#include <algorithm>
#include <cstring>
#include <deque>
#include <limits>
#include <set>
#include <string_view>

#include "Core/TensorDispatch.h"
#include "IO/ZipArchive.h"

namespace elman::io {
namespace {

enum class Kind { None, Integer, Boolean, Scalar, String, Global, Tuple, List, Map, Storage, Tensor, Object, Mark };

struct Value {
    Kind kind;
    int64_t integer = 0;
    double scalar = 0;
    std::string text;
    std::vector<Value*> sequence;
    std::map<std::string, Value*> mapping;
    DType dtype = DType::Float32;
    bool typed = false;
};

bool suffix(std::string_view value, std::string_view end) {
    return value.size() >= end.size() && value.substr(value.size() - end.size()) == end;
}

DType torch_dtype(const Value& type) {
    require(type.kind == Kind::Global && type.text.rfind("torch ", 0) == 0, "Unsupported weight storage type");
    const auto name = std::string_view(type.text).substr(6);
    static const std::map<std::string_view, DType> types{
        {"FloatStorage", DType::Float32}, {"float32", DType::Float32}, {"HalfStorage", DType::Float16}, {"float16", DType::Float16},
        {"DoubleStorage", DType::Float64}, {"float64", DType::Float64}, {"BFloat16Storage", DType::BFloat16}, {"bfloat16", DType::BFloat16},
        {"ByteStorage", DType::UInt8}, {"uint8", DType::UInt8}, {"CharStorage", DType::Int8}, {"int8", DType::Int8},
        {"ShortStorage", DType::Int16}, {"int16", DType::Int16}, {"IntStorage", DType::Int32}, {"int32", DType::Int32},
        {"LongStorage", DType::Int64}, {"int64", DType::Int64}, {"BoolStorage", DType::Bool}, {"bool", DType::Bool},
        {"ComplexFloatStorage", DType::Complex64}, {"complex64", DType::Complex64},
        {"ComplexDoubleStorage", DType::Complex128}, {"complex128", DType::Complex128},
        {"complex32", DType::Complex32},
        {"float8_e4m3fn", DType::Float8E4M3FN}, {"float8_e5m2", DType::Float8E5M2},
        {"uint16", DType::UInt16}, {"uint32", DType::UInt32}, {"uint64", DType::UInt64}
    };
    const auto found = types.find(name);
    require(found != types.end(), "Unsupported weight storage type: " + std::string(name));
    return found->second;
}

int64_t nonnegative(const Value& value) {
    require(value.kind == Kind::Integer && value.integer >= 0, "Expected a nonnegative weight index");
    return value.integer;
}

// Decode a restricted data language. Globals are labels for the tensor
// constructors below. No file-supplied class or callable is executed.
class WeightPickle {
public:
    explicit WeightPickle(Bytes bytes, bool scripted = false) : bytes_(bytes), scripted_(scripted) {}
    const Value& parse() {
        while (position_ < bytes_.size) {
            switch (const auto opcode = read(1); opcode) {
            case 0x80: { const auto version = read(1); require(version >= 2 && version <= 5, "Unsupported pickle protocol"); break; }
            case 0x95: { const auto size = read(8); require(size <= bytes_.size - position_, "Truncated pickle frame"); break; }
            case '.': require(stack_.size() == 1 && position_ == bytes_.size, "Malformed weight pickle result"); return *stack_.back();
            case '(': stack_.push_back(make(Kind::Mark)); break;
            case 'N': stack_.push_back(make(Kind::None)); break;
            case 0x88: case 0x89: {
                auto* value = make(Kind::Boolean);
                value->integer = opcode == 0x88;
                stack_.push_back(value);
                break;
            }
            case 'G': {
                uint64_t bits = 0;
                for (int byte = 0; byte < 8; ++byte) bits = (bits << 8) | read(1);
                auto* value = make(Kind::Scalar);
                std::memcpy(&value->scalar, &bits, sizeof(bits));
                stack_.push_back(value);
                break;
            }
            case 'K': integer(static_cast<int64_t>(read(1))); break;
            case 'M': integer(static_cast<int64_t>(read(2))); break;
            case 'J': integer(signed_number(4)); break;
            case 0x8a: integer(signed_number(static_cast<size_t>(read(1)))); break;
            case 0x8b: integer(signed_number(static_cast<size_t>(read(4)))); break;
            case 'X': case 'T': string(read(4)); break;
            case 'U': case 0x8c: string(read(1)); break;
            case 0x8d: string(read(8)); break;
            case ')': stack_.push_back(make(Kind::Tuple)); break;
            case ']': stack_.push_back(make(Kind::List)); break;
            case '}': stack_.push_back(make(Kind::Map)); break;
            case 't': sequence(Kind::Tuple, marked()); break;
            case 'l': sequence(Kind::List, marked()); break;
            case 'd': {
                auto pairs = marked();
                auto* value = make(Kind::Map);
                assign(*value, pairs);
                stack_.push_back(value);
                break;
            }
            case 0x85: case 0x86: case 0x87: sequence(Kind::Tuple, take(static_cast<size_t>(opcode - 0x84))); break;
            case 'a': { auto* item = pop(); append(top(), {item}); break; }
            case 'e': { auto items = marked(); append(top(), items); break; }
            case 's': { auto items = take(2); assign(top(), items); break; }
            case 'u': { auto items = marked(); assign(top(), items); break; }
            case 'q': memo_[read(1)] = &top(); break;
            case 'r': memo_[read(4)] = &top(); break;
            case 0x94: { const auto index = memo_.size(); memo_[index] = &top(); break; }
            case 'h': recall(read(1)); break;
            case 'j': recall(read(4)); break;
            case 'c': { auto module = line(); global(std::move(module), line()); break; }
            case 0x93: {
                const auto* name = pop();
                const auto* module = pop();
                require(name->kind == Kind::String && module->kind == Kind::String, "Invalid weight GLOBAL label");
                global(module->text, name->text);
                break;
            }
            case 'Q': persistent(); break;
            case 'R': reduce(); break;
            case 0x81:
                if (scripted_) new_object();
                else reduce();
                break;
            case 'b': {
                const auto* metadata = pop();
                if (scripted_ && top().kind == Kind::Object) {
                    require(metadata->kind == Kind::Map || metadata->kind == Kind::None,
                            "Unsupported TorchScript serialized state");
                    top().mapping = metadata->mapping;
                } else require(top().kind == Kind::Map && (metadata->kind == Kind::Map || metadata->kind == Kind::None),
                               "Unsupported serialized weight metadata");
                break;
            }
            default: throw std::invalid_argument("Unsupported weight pickle opcode " + std::to_string(opcode));
            }
        }
        throw std::invalid_argument("Weight pickle has no terminator");
    }

private:
    uint64_t read(size_t width) {
        const auto value = bytes_.le(position_, width);
        position_ += width;
        return value;
    }
    int64_t signed_number(size_t width) {
        require(width <= 8, "Integer in weight pickle is too wide");
        if (!width) return 0;
        auto bits = read(width);
        if (width < 8 && (bits & (uint64_t(1) << (width * 8 - 1)))) bits |= (~uint64_t(0) << (width * 8));
        int64_t value;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }
    void integer(int64_t value) {
        auto* result = make(Kind::Integer);
        result->integer = value;
        stack_.push_back(result);
    }
    void string(uint64_t length) {
        require(length <= bytes_.size - position_, "Truncated string in weight pickle");
        auto* result = make(Kind::String);
        result->text = bytes_.text(position_, static_cast<size_t>(length));
        position_ += static_cast<size_t>(length);
        stack_.push_back(result);
    }
    std::string line() {
        const auto start = position_;
        while (position_ < bytes_.size && bytes_.data[position_] != '\n') ++position_;
        require(position_ < bytes_.size, "Truncated GLOBAL label in weight pickle");
        return bytes_.text(start, position_++ - start);
    }
    Value* make(Kind kind) {
        values_.push_back(Value{kind});
        return &values_.back();
    }
    Value& top() {
        require(!stack_.empty(), "Empty weight pickle stack");
        return *stack_.back();
    }
    Value* pop() { auto* value = &top(); stack_.pop_back(); return value; }
    std::vector<Value*> take(size_t size) {
        require(size <= stack_.size(), "Truncated weight pickle stack");
        const auto first = stack_.end() - size;
        std::vector<Value*> result(first, stack_.end());
        stack_.erase(first, stack_.end());
        return result;
    }
    std::vector<Value*> marked() {
        const auto marker = std::find_if(stack_.rbegin(), stack_.rend(), [](Value* value) { return value->kind == Kind::Mark; });
        require(marker != stack_.rend(), "Missing weight pickle mark");
        auto items = take(static_cast<size_t>(std::distance(stack_.rbegin(), marker)));
        pop();
        return items;
    }
    void sequence(Kind kind, std::vector<Value*> items) {
        auto* result = make(kind);
        result->sequence = std::move(items);
        stack_.push_back(result);
    }
    void assign(Value& value, const std::vector<Value*>& pairs) {
        require(value.kind == Kind::Map && pairs.size() % 2 == 0, "Invalid weight pickle mapping");
        for (size_t i = 0; i < pairs.size(); i += 2) {
            const auto& key = *pairs[i];
            require(key.kind == Kind::String || key.kind == Kind::Integer, "Unsupported weight pickle key type");
            value.mapping[key.kind == Kind::String ? key.text : std::to_string(key.integer)] = pairs[i + 1];
        }
    }
    void append(Value& value, const std::vector<Value*>& items) {
        require(value.kind == Kind::List, "Invalid weight pickle list");
        value.sequence.insert(value.sequence.end(), items.begin(), items.end());
    }
    void recall(uint64_t index) {
        const auto found = memo_.find(index);
        require(found != memo_.end(), "Unknown weight pickle memo index");
        stack_.push_back(found->second);
    }
    void global(std::string module, const std::string& name) {
        auto* value = make(Kind::Global);
        value->text = std::move(module) + ' ' + name;
        stack_.push_back(value);
    }
    void persistent() {
        const auto* value = pop();
        const auto& fields = value->sequence;
        require(value->kind == Kind::Tuple && fields.size() == 5 && fields[0]->kind == Kind::String && fields[0]->text == "storage" &&
                fields[1]->kind == Kind::Global && fields[2]->kind == Kind::String && fields[3]->kind == Kind::String,
                "Unsupported persistent weight storage record");
        auto* storage = make(Kind::Storage);
        storage->integer = nonnegative(*fields[4]);
        storage->text = fields[2]->text;
        require(!storage->text.empty() && std::all_of(storage->text.begin(), storage->text.end(),
                [](char c) { return c >= '0' && c <= '9'; }), "Invalid weight storage key");
        storage->typed = fields[1]->text != "torch.storage UntypedStorage" && fields[1]->text != "torch UntypedStorage";
        if (storage->typed) storage->dtype = torch_dtype(*fields[1]);
        stack_.push_back(storage);
    }
    void new_object() {
        const auto* arguments = pop();
        const auto* type = pop();
        require(type->kind == Kind::Global && type->text.rfind("__torch__", 0) == 0 &&
                arguments->kind == Kind::Tuple && arguments->sequence.empty(), "Unsupported TorchScript class constructor");
        const auto separator = type->text.find(' ');
        require(separator != std::string::npos && type->text.find(' ', separator + 1) == std::string::npos,
                "Invalid TorchScript class reference");
        auto* object = make(Kind::Object);
        object->text = type->text.substr(0, separator) + '.' + type->text.substr(separator + 1);
        stack_.push_back(object);
    }
    void reduce() {
        auto* args = pop();
        const auto* name = pop();
        require(name->kind == Kind::Global && args->kind == Kind::Tuple, "Invalid weight pickle constructor");
        const auto& values = args->sequence;
        if (scripted_ && (name->text == "torch.jit._pickle build_intlist" ||
                          name->text == "torch.jit._pickle build_tensorlist" ||
                          name->text == "torch.jit._pickle build_doublelist" ||
                          name->text == "torch.jit._pickle build_boollist" ||
                          name->text == "torch.jit._pickle restore_type_tag")) {
            require(!values.empty() && values.size() <= 2, "Invalid TorchScript typed state");
            stack_.push_back(values[0]);
            return;
        }
        if (name->text == "collections OrderedDict") {
            auto* map = make(Kind::Map);
            require(values.size() <= 1 && (values.empty() || values[0]->kind == Kind::List || values[0]->kind == Kind::Tuple),
                    "Invalid serialized ordered dictionary");
            if (!values.empty()) for (const auto* pair : values[0]->sequence) {
                require(pair->kind == Kind::Tuple && pair->sequence.size() == 2, "Invalid serialized ordered dictionary entry");
                assign(*map, pair->sequence);
            }
            stack_.push_back(map);
            return;
        }
        if (name->text == "torch._utils _rebuild_parameter" || name->text == "torch._utils _rebuild_parameter_with_state") {
            require(!values.empty() && values[0]->kind == Kind::Tensor, "Invalid serialized parameter");
            stack_.push_back(values[0]);
            return;
        }
        const bool v3 = name->text == "torch._utils _rebuild_tensor_v3";
        require(v3 || name->text == "torch._utils _rebuild_tensor" || name->text == "torch._utils _rebuild_tensor_v2",
                "Unsupported weight pickle constructor: " + name->text);
        require(values.size() >= (v3 ? 7 : 4) && values[0]->kind == Kind::Storage && values[2]->kind == Kind::Tuple &&
                values[3]->kind == Kind::Tuple && (v3 || values[0]->typed), "Invalid serialized weight tensor");
        auto* tensor = make(Kind::Tensor);
        tensor->dtype = v3 ? torch_dtype(*values[6]) : values[0]->dtype;
        tensor->integer = nonnegative(*values[1]);
        tensor->sequence = {values[0], values[2], values[3]};
        stack_.push_back(tensor);
    }

    Bytes bytes_;
    size_t position_ = 0;
    std::deque<Value> values_;
    std::vector<Value*> stack_;
    std::map<uint64_t, Value*> memo_;
    bool scripted_;
};

Tensor materialize(ZipArchive& file, const std::string& storage_directory, const Value& value, bool big_endian, Backend backend) {
    const auto& storage = *value.sequence[0];
    const auto& dimensions = value.sequence[1]->sequence;
    const auto& strides = value.sequence[2]->sequence;
    require(dimensions.size() <= 8 && dimensions.size() == strides.size(), "Unexpected weight tensor rank or strides");
    Shape shape;
    for (const auto* dim : dimensions) {
        const auto size = nonnegative(*dim);
        require(size <= std::numeric_limits<int>::max(), "Weight dimension exceeds the Elman indexing limit");
        shape.push_back(static_cast<int>(size));
    }
    const auto count = shape_element_count(shape);
    const size_t width = dtype_info(value.dtype).bits / 8;
    const auto name = storage_directory + storage.text;
    const auto entry = file.entries().find(name);
    const auto quantity = static_cast<uint64_t>(storage.integer);
    require(entry != file.entries().end() && (!storage.typed || storage.dtype == value.dtype) &&
            (!storage.typed || quantity <= std::numeric_limits<uint64_t>::max() / width) &&
            entry->second.size == quantity * (storage.typed ? width : 1u) && entry->second.size % width == 0,
            "Weight storage size differs from its metadata");
    const uint64_t elements = entry->second.size / width;
    const auto offset = static_cast<uint64_t>(value.integer);
    require(offset <= elements && (!count || offset < elements), "Weight tensor starts outside storage");
    std::vector<uint64_t> steps;
    uint64_t last = offset;
    for (size_t axis = 0; axis < strides.size(); ++axis) {
        const auto step = static_cast<uint64_t>(nonnegative(*strides[axis]));
        steps.push_back(step);
        if (count && shape[axis] > 1) {
            require(step <= (elements - 1 - last) / static_cast<uint64_t>(shape[axis] - 1), "Weight tensor view exceeds storage");
            last += step * (shape[axis] - 1);
        }
    }
    const auto data = file.read(name);
    std::vector<uint8_t> payload(dtype_storage_bytes(value.dtype, count));
    if (count) {
        size_t stride = 1;
        bool contiguous = true;
        for (size_t axis = shape.size(); axis > 0;) {
            --axis;
            if (shape[axis] > 1 && steps[axis] != stride) contiguous = false;
            stride *= shape[axis];
        }
        if (contiguous) {
            std::copy_n(data.data() + offset * width, payload.size(), payload.data());
        } else {
            for (size_t i = 0; i < count; ++i) {
                uint64_t index = offset;
                size_t remaining = i;
                for (size_t axis = shape.size(); axis > 0;) {
                    --axis;
                    const auto dim = static_cast<size_t>(shape[axis]);
                    index += (remaining % dim) * steps[axis];
                    remaining /= dim;
                }
                std::copy_n(data.data() + index * width, width, payload.data() + i * width);
            }
        }
    }
    if (big_endian) {
        const auto component = is_complex(value.dtype) ? width / 2 : width;
        for (size_t i = 0; i < payload.size(); i += component) std::reverse(payload.begin() + i, payload.begin() + i + component);
    }
    return Tensor::from_raw(shape, value.dtype, payload, backend);
}

}  // namespace

TensorArchive torch_weights(const std::filesystem::path& path, Backend backend) {
    ZipArchive file(path);
    std::string pickle_path;
    for (const auto& [name, entry] : file.entries()) {
        if (name == "data.pkl" || suffix(name, "/data.pkl")) {
            require(pickle_path.empty(), "Multiple weight pickle records in the same archive");
            pickle_path = name;
        }
    }
    require(!pickle_path.empty(), "Weight ZIP archive is missing data.pkl");
    const auto root = pickle_path.substr(0, pickle_path.size() - std::string_view("data.pkl").size());
    require(!file.entries().count(root + "constants.pkl"), "TorchScript graphs are outside the weight archive format");
    bool big_endian = false;
    if (file.entries().count(root + "byteorder")) {
        const auto bytes = file.read(root + "byteorder");
        const auto order = view(bytes).text(0, bytes.size());
        require(order == "little" || order == "big", "Unknown weight archive byte order");
        big_endian = order == "big";
    }
    const auto bytes = file.read(pickle_path);
    WeightPickle pickle(view(bytes));
    const Value* weights = &pickle.parse();
    require(weights->kind == Kind::Map, "Expected a tensor state dictionary at the root of the weight archive");
    for (const auto* key : {"state_dict", "model_state_dict", "model"}) {
        const auto found = weights->mapping.find(key);
        if (found != weights->mapping.end() && found->second->kind == Kind::Map) { weights = found->second; break; }
    }
    TensorArchive result;
    for (const auto& [name, value] : weights->mapping) {
        if (name == "_metadata" && value->kind == Kind::Map) continue;
        require(value->kind == Kind::Tensor && !name.empty(), "State dictionary entry is not a named tensor: " + name);
        result.emplace(name, materialize(file, root + "data/", *value, big_endian, backend));
    }
    require(!result.empty(), "State dictionary has no tensors");
    return result;
}

TorchScriptArchive::TorchScriptArchive(const std::filesystem::path& path, Backend backend) {
    ZipArchive file(path);
    std::string root;
    bool found_root = false;
    for (const auto& entry : file.entries()) {
        const auto& name = entry.first;
        if (name == "constants.pkl" || suffix(name, "/constants.pkl")) {
            require(!found_root, "Ambiguous TorchScript archive root");
            root = name.substr(0, name.size() - std::string_view("constants.pkl").size());
            found_root = true;
        }
    }
    require(found_root && file.entries().count(root + "data.pkl"), "Missing TorchScript graph archive records");
    for (const auto& entry : file.entries()) {
        const auto& name = entry.first;
        if (name.rfind(root + "code/", 0) != 0 || !suffix(name, ".py")) continue;
        const auto bytes = file.read(name);
        source.emplace(name.substr(root.size() + 5), view(bytes).text(0, bytes.size()));
    }
    require(!source.empty(), "TorchScript archive does not contain graph source");
    bool big_endian = false;
    if (file.entries().count(root + "byteorder")) {
        const auto bytes = file.read(root + "byteorder");
        const auto order = view(bytes).text(0, bytes.size());
        require(order == "little" || order == "big", "Unknown TorchScript storage byte order");
        big_endian = order == "big";
    }
    const auto read_state = [&](const std::string& record) {
        const auto bytes = file.read(root + record + ".pkl");
        WeightPickle pickle(view(bytes), true);
        std::map<const Value*, std::shared_ptr<const TorchScriptValue>> cache;
        std::set<const Value*> active;
        auto convert = [&](auto&& convert, const Value& value, unsigned depth) -> std::shared_ptr<const TorchScriptValue> {
            using Output = TorchScriptValue::Kind;
            if (const auto cached = cache.find(&value); cached != cache.end()) return cached->second;
            require(depth < 128 && active.insert(&value).second, "Recursive TorchScript module state");
            auto result = std::make_shared<TorchScriptValue>();
            switch (value.kind) {
                case Kind::None: break;
                case Kind::Boolean: result->kind = Output::Boolean; result->integer = value.integer; break;
                case Kind::Integer: result->kind = Output::Integer; result->integer = value.integer; break;
                case Kind::Scalar: result->kind = Output::Number; result->number = value.scalar; break;
                case Kind::String: result->kind = Output::String; result->text = value.text; break;
                case Kind::Tensor:
                    result->kind = Output::Tensor;
                    result->tensor = materialize(file, root + record + '/', value, big_endian, backend);
                    break;
                case Kind::Tuple: case Kind::List:
                    result->kind = Output::List;
                    for (const auto* item : value.sequence) result->sequence.push_back(convert(convert, *item, depth + 1));
                    break;
                case Kind::Object: case Kind::Map:
                    result->kind = Output::Object;
                    result->text = value.text;
                    for (const auto& [name, item] : value.mapping) result->fields.emplace(name, convert(convert, *item, depth + 1));
                    break;
                default: throw std::invalid_argument("Unsupported TorchScript stored value");
            }
            active.erase(&value);
            cache.emplace(&value, result);
            return result;
        };
        return convert(convert, pickle.parse(), 0);
    };
    module = read_state("data");
    constants = read_state("constants");
    require(module->kind == TorchScriptValue::Kind::Object && constants->kind == TorchScriptValue::Kind::List,
            "Invalid TorchScript module or constant table");
}

}  // namespace elman::io
