#include "Inference/Importers/TorchScript.h"
#include "Core/LocalFunction.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <locale>
#include <optional>
#include <set>
#include <sstream>
#include <string_view>

#include "Core/Device.h"
#include "IO/TorchScriptArchive.h"
#include "Inference/GraphBuilder.h"

namespace elman {
namespace {

using io::require;
using io::TorchScriptValue;
using namespace std::string_view_literals;

struct Line { int indent; std::string code; };

std::string_view trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return text;
}

std::vector<Line> lines(std::string_view source) {
    std::vector<Line> out;
    int balance = 0, indent = 0;
    std::string statement;
    while (!source.empty()) {
        const size_t newline = source.find('\n');
        const std::string_view line = source.substr(0, newline);
        source.remove_prefix(newline == source.npos ? source.size() : newline + 1);
        const std::string_view token = trim(line);
        if (token.empty() || token.front() == '#') continue;
        if (statement.empty()) indent = int(token.data() - line.data());
        char quote = 0;
        bool escaped = false;
        size_t length = 0;
        for (; length < token.size(); ++length) {
            const char c = token[length];
            if (escaped) escaped = false;
            else if (quote) {
                if (c == '\\') escaped = true;
                else if (c == quote) quote = 0;
            } else if (c == '\'' || c == '"') quote = c;
            else if (c == '#') break;
            else if (c == '(' || c == '[' || c == '{') ++balance;
            else if (c == ')' || c == ']' || c == '}') --balance;
        }
        require(!quote && balance >= 0 && balance <= 128, "Invalid serialized TorchScript source");
        if (!statement.empty()) statement += ' ';
        statement += token.substr(0, length);
        if (!balance) {
            out.push_back({indent, std::string(trim(statement))});
            statement.clear();
        }
    }
    require(statement.empty(), "Truncated TorchScript source statement");
    return out;
}

struct Method {
    std::vector<std::string> parameters;
    std::vector<Line> body;
};

std::vector<std::string> parameter_names(std::string_view header) {
    const size_t open = header.find('(');
    require(open != header.npos, "Missing TorchScript function parameters");
    std::vector<std::string> names;
    size_t start = open + 1;
    int depth = 0;
    for (size_t i = start; i < header.size(); ++i) {
        const char c = header[i];
        if (!depth && (c == ',' || c == ')')) {
            const std::string_view parameter = trim(header.substr(start, i - start));
            const size_t end = parameter.find_first_of(":=");
            if (!parameter.empty()) names.emplace_back(trim(parameter.substr(0, end)));
            start = i + 1;
            if (c == ')') return names;
        } else if (c == '[' || c == '(' || c == '{') ++depth;
        else if (c == ']' || c == ')' || c == '}') --depth;
    }
    throw std::invalid_argument("Unterminated TorchScript parameter list");
}

std::map<std::string, Method> methods(const std::map<std::string, std::string>& sources) {
    std::map<std::string, Method> result;
    for (const std::pair<const std::string, std::string>& entry : sources) {
        const std::string& path = entry.first;
        const std::string& source = entry.second;
        std::string package = path.substr(0, path.size() - 3);
        std::replace(package.begin(), package.end(), '/', '.');
        const std::vector<Line> statements = lines(source);
        std::string type;
        for (size_t i = 0; i < statements.size();) {
            const Line& statement = statements[i];
            if (statement.code.rfind("class ", 0) == 0 && statement.indent == 0) {
                const size_t class_end = statement.code.find_first_of("(:", 6);
                type = std::string(trim(std::string_view(statement.code).substr(6, class_end - 6)));
            }
            if (statement.code.rfind("def ", 0) != 0) { ++i; continue; }
            const size_t open = statement.code.find('(');
            require(open != std::string::npos, "Missing TorchScript function name");
            const std::string prefix = statement.indent == 0 ? package : package + '.' + type;
            const std::string name = prefix + '.' + std::string(trim(std::string_view(statement.code).substr(4, open - 4)));
            Method method{parameter_names(statement.code), {}};
            ++i;
            while (i < statements.size() && statements[i].indent > statement.indent) method.body.push_back(statements[i++]);
            require(!method.body.empty() && result.emplace(name, std::move(method)).second, "Invalid TorchScript function " + name);
        }
    }
    return result;
}

enum class Kind { None, Boolean, Integer, Number, String, Tensor, Sequence, Mapping, Module, Symbol, Callable };

struct Value {
    Kind kind = Kind::None;
    std::string text;
    int64_t integer = 0;
    double number = 0;
    const TorchScriptValue* module = nullptr;
    std::vector<Value> elements;
    std::map<std::string, Value> fields;
};

using Scope = std::map<std::string, Value>;

Value integer(int64_t value) { Value out{Kind::Integer}; out.integer = value; return out; }
Value number(double value) { Value out{Kind::Number}; out.number = value; return out; }
Value boolean(bool value) { Value out{Kind::Boolean}; out.integer = value; return out; }
Value module(const TorchScriptValue& value) { Value out{Kind::Module}; out.module = &value; return out; }

double numeric(const Value& value) {
    if (value.kind == Kind::Number) return value.number;
    require(value.kind == Kind::Integer || value.kind == Kind::Boolean, "TorchScript expected a numeric argument");
    return double(value.integer);
}

int integer_argument(const Value& value) {
    require(value.kind == Kind::Integer && value.integer >= INT32_MIN && value.integer <= INT32_MAX,
            "TorchScript expected a representable integer argument");
    return int(value.integer);
}

bool truth(const Value& value) {
    require(value.kind == Kind::Boolean || value.kind == Kind::Integer, "TorchScript condition must resolve to a constant");
    return value.integer != 0;
}

class Compiler {
public:
    Compiler(GraphBuilder& graph, const io::TorchScriptArchive& archive)
        : graph_(graph), archive_(archive), methods_(methods(archive.source)) {}

    const Method& forward() const { return method(archive_.module->text + ".forward"); }
    Value run(std::vector<Value> inputs) {
        inputs.insert(inputs.begin(), module(*archive_.module));
        return invoke(archive_.module->text + ".forward", inputs, 0);
    }
    std::string current(const Value& value) const {
        std::string name = value.text;
        for (auto next = updated_.find(name); next != updated_.end(); next = updated_.find(name))
            name = next->second;
        return name;
    }

    Value member(const Value& object, const std::string& name) {
        if (object.kind == Kind::Module) {
            const auto found = object.module->fields.find(name);
            if (found != object.module->fields.end()) return stored(*found->second);
            Value result{Kind::Callable, object.module->text + '.' + name};
            result.module = object.module;
            return result;
        }
        if (object.kind == Kind::Tensor) {
            Value result{Kind::Callable, "torch." + name};
            result.elements = {object};
            return result;
        }
        require(object.kind == Kind::Symbol, "Unsupported TorchScript member access");
        if (object.text == "CONSTANTS" && name.size() > 1 && name[0] == 'c') {
            size_t index = 0;
            const std::from_chars_result parsed = std::from_chars(name.data() + 1, name.data() + name.size(), index);
            require(parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() &&
                    index < archive_.constants->sequence.size(), "Invalid TorchScript constant reference");
            return stored(*archive_.constants->sequence[index]);
        }
        return {Kind::Symbol, object.text + '.' + name};
    }

    Value call(Value callee, std::vector<Value> args, const Scope& keywords, size_t depth) {
        require(callee.kind == Kind::Symbol || callee.kind == Kind::Callable, "TorchScript expression is not callable");
        if (callee.module) args.insert(args.begin(), module(*callee.module));
        else if (!callee.elements.empty()) args.insert(args.begin(), callee.elements.front());
        if (methods_.count(callee.text)) {
            require(keywords.empty(), "TorchScript function keyword binding is unsupported");
            return invoke(callee.text, args, depth + 1);
        }
        return builtin(callee.text, args, keywords);
    }

    Value binary(const std::string& op, const Value& a, const Value& b) {
        if (a.kind == Kind::Tensor || b.kind == Kind::Tensor) {
            const std::string operation = op == "Div" ? "TrueDivide" : op;
            if (a.kind == Kind::Tensor && b.kind == Kind::Tensor) return emit(operation, {a, b});
            const bool reverse = a.kind != Kind::Tensor;
            const Value& literal = reverse ? a : b;
            Json::Object attrs{{"op", operation}, {"reverse", reverse}};
            if (literal.kind == Kind::Integer || literal.kind == Kind::Boolean) {
                attrs.emplace("integer", std::to_string(literal.integer));
                attrs.emplace("boolean", literal.kind == Kind::Boolean);
            } else attrs.emplace("value", numeric(literal));
            return emit("ScalarBinary", {reverse ? b : a}, std::move(attrs));
        }
        const double x = numeric(a), y = numeric(b);
        if (op == "Eq") return boolean(x == y);
        if (op == "Ne") return boolean(x != y);
        if (op == "Gt") return boolean(x > y);
        if (op == "Ge") return boolean(x >= y);
        if (op == "Lt") return boolean(x < y);
        if (op == "Le") return boolean(x <= y);
        const double result = op == "Add" ? x + y : op == "Sub" ? x - y : op == "Mul" ? x * y : x / y;
        if (a.kind == Kind::Integer && b.kind == Kind::Integer && op != "Div") {
            require(result >= -0x1p63 && result < 0x1p63, "TorchScript integer arithmetic overflow");
            return integer(int64_t(result));
        }
        return number(result);
    }

private:
    class Expression;

    const Method& method(const std::string& key) const {
        const auto found = methods_.find(key);
        require(found != methods_.end(), "Missing TorchScript method " + key);
        return found->second;
    }
    Value invoke(const std::string& key, const std::vector<Value>& args, size_t depth);
    std::optional<Value> block(const Method& source, Scope& scope, size_t& position, size_t end, int indent, size_t depth);

    Value stored(const TorchScriptValue& value) {
        using Storage = TorchScriptValue::Kind;
        switch (value.kind) {
            case Storage::None: return {};
            case Storage::Boolean: return boolean(value.integer != 0);
            case Storage::Integer: return integer(value.integer);
            case Storage::Number: return number(value.number);
            case Storage::String: return {Kind::String, value.text};
            case Storage::Object: return module(value);
            case Storage::List: {
                Value result{Kind::Sequence};
                for (const std::shared_ptr<const io::TorchScriptValue>& item : value.sequence) result.elements.push_back(stored(*item));
                return result;
            }
            case Storage::Tensor: {
                auto found = tensors_.find(&value);
                if (found == tensors_.end()) found = tensors_.emplace(&value, graph_.constant(value.tensor)).first;
                return {Kind::Tensor, found->second};
            }
        }
        throw std::invalid_argument("Unsupported TorchScript state value");
    }

    Value emit(std::string op, const std::vector<Value>& args, Json::Object attrs = {}) {
        std::vector<std::string> operands;
        for (const Value& value : args) {
            require(value.kind == Kind::Tensor || value.kind == Kind::None, "TorchScript operation requires tensor operands: " + op);
            operands.push_back(current(value));
        }
        static const std::set<std::string> views{
            "Reshape", "Transpose", "SwapAxes", "ExpandDims", "Squeeze", "FlattenRange", "TensorIndex",
        };
        const bool alias = views.count(op) != 0;
        std::string output = graph_.operation(std::move(op), operands, std::move(attrs));
        if (alias) {
            view_values_.insert(operands.front());
            view_values_.insert(output);
        }
        return {Kind::Tensor, std::move(output)};
    }

    Value functionalize(const Value& value, const Value& replacement) {
        const std::string previous = current(value);
        require(value.kind == Kind::Tensor && replacement.kind == Kind::Tensor &&
                !graph_.constant_value(previous) && !view_values_.count(previous),
                "Unsupported TorchScript mutation of a constant or tensor view");
        require(updated_.emplace(previous, replacement.text).second, "Duplicate TorchScript tensor update");
        return value;
    }

    Value reshape(const Value& data, const Value& dimensions) {
        if (dimensions.kind == Kind::Tensor) return emit("Reshape", {data, dimensions}, {{"allowzero", 1}});
        require(dimensions.kind == Kind::Sequence, "TorchScript reshape requires dimensions");
        std::vector<Value> runtime;
        Json::Array shape;
        for (const Value& dim : dimensions.elements) {
            if (dim.kind == Kind::Tensor) {
                runtime.push_back(emit("Cast", {dim}, {{"to", dtype_name(DType::Int64)}}));
                continue;
            }
            const int value = integer_argument(dim);
            shape.emplace_back(value);
            runtime.push_back({Kind::Tensor, graph_.constant(Tensor::from_int64_data({}, {value}, graph_.backend()))});
        }
        return shape.size() == dimensions.elements.size() ? emit("Reshape", {data}, {{"shape", std::move(shape)}, {"allowzero", 1}}) :
               emit("Reshape", {data, emit("Stack", runtime)}, {{"allowzero", 1}});
    }

    Value builtin(const std::string& name, const std::vector<Value>& args, const Scope& keywords);

    GraphBuilder& graph_;
    const io::TorchScriptArchive& archive_;
    std::map<std::string, Method> methods_;
    std::map<const TorchScriptValue*, std::string> tensors_;
    std::map<std::string, std::string> updated_;
    std::set<std::string> view_values_;
};

class Compiler::Expression {
public:
    Expression(Compiler& compiler, const Scope& scope, std::string_view text, size_t depth)
        : compiler_(compiler), scope_(scope), text_(text), depth_(depth) {}
    Value parse() {
        Value result = expression();
        require(trim(text_).empty(), "Unsupported TorchScript expression syntax near " + std::string(text_));
        return result;
    }

private:
    bool take(std::string_view token) {
        text_ = trim(text_);
        if (text_.substr(0, token.size()) != token) return false;
        text_.remove_prefix(token.size());
        return true;
    }
    void expect(std::string_view token) {
        require(take(token), "Expected TorchScript token " + std::string(token));
    }
    std::string identifier() {
        text_ = trim(text_);
        size_t count = 0;
        while (count < text_.size() && (std::isalnum(static_cast<unsigned char>(text_[count])) || text_[count] == '_')) ++count;
        require(count, "Expected a TorchScript identifier");
        std::string name(text_.substr(0, count));
        text_.remove_prefix(count);
        return name;
    }
    Value expression(int minimum = 0) {
        require(depth_ + nesting_ < 128, "TorchScript expression nesting exceeds the importer limit");
        ++nesting_;
        Value left = primary();
        struct Binary { std::string_view code; const char* op; int precedence; };
        static const Binary tokens[]{
            {"==", "Eq", 1}, {"!=", "Ne", 1}, {">=", "Ge", 1}, {"<=", "Le", 1},
            {">", "Gt", 1}, {"<", "Lt", 1}, {"+", "Add", 2}, {"-", "Sub", 2}, {"*", "Mul", 3}, {"/", "Div", 3}
        };
        while (true) {
            const Binary* const op = std::find_if(std::begin(tokens), std::end(tokens),
                             [&](const Binary& token) { return token.precedence >= minimum && take(token.code); });
            if (op == std::end(tokens)) break;
            left = compiler_.binary(op->op, left, expression(op->precedence + 1));
        }
        --nesting_;
        return left;
    }
    Value quoted() {
        const char quote = text_.front();
        text_.remove_prefix(1);
        std::string value;
        while (!text_.empty()) {
            char ch = text_.front();
            text_.remove_prefix(1);
            if (ch == quote) return {Kind::String, std::move(value)};
            if (ch == '\\') {
                require(!text_.empty(), "Truncated TorchScript string escape");
                ch = text_.front();
                text_.remove_prefix(1);
                if (ch == 'n') ch = '\n';
                else if (ch == 't') ch = '\t';
                else if (ch == 'r') ch = '\r';
                else require(ch == '"' || ch == '\'' || ch == '\\', "Unsupported TorchScript string escape");
            }
            value += ch;
        }
        throw std::invalid_argument("Unterminated TorchScript string");
    }
    Value scalar() {
        size_t at = 0;
        LocalFunction digits{[&] { while (at < text_.size() && std::isdigit(static_cast<unsigned char>(text_[at]))) ++at; }};
        digits();
        bool floating = false;
        if (at < text_.size() && text_[at] == '.') { ++at; digits(); floating = true; }
        if (at < text_.size() && (text_[at] == 'e' || text_[at] == 'E')) {
            ++at;
            if (at < text_.size() && (text_[at] == '+' || text_[at] == '-')) ++at;
            digits();
            floating = true;
        }
        const std::string_view token = text_.substr(0, at);
        text_.remove_prefix(at);
        if (floating) {
            std::istringstream in{std::string(token)};
            in.imbue(std::locale::classic());
            double value = 0;
            in >> value;
            require(in && in.peek() == std::char_traits<char>::eof(), "Invalid TorchScript floating literal");
            return number(value);
        }
        int64_t value = 0;
        const std::from_chars_result parsed = std::from_chars(token.data(), token.data() + token.size(), value);
        require(parsed.ec == std::errc{} && parsed.ptr == token.data() + token.size(), "Invalid TorchScript integer literal");
        return integer(value);
    }
    Value atom() {
        text_ = trim(text_);
        require(!text_.empty(), "Missing TorchScript expression");
        if (take("-")) return compiler_.binary("Sub", integer(0), expression(4));
        if (take("(") || take("[")) {
            const bool tuple = last_delimiter_ == '(';
            return sequence(tuple);
        }
        if (take("{")) {
            Value result{Kind::Mapping};
            while (!take("}")) {
                Value key = expression();
                expect(":");
                require(key.kind == Kind::String && result.fields.emplace(key.text, expression()).second,
                        "TorchScript graph output dictionaries require distinct string keys");
                if (!take(",")) { expect("}"); break; }
            }
            return result;
        }
        if (text_.front() == '"' || text_.front() == '\'') return quoted();
        if (std::isdigit(static_cast<unsigned char>(text_.front()))) return scalar();
        const std::string name = identifier();
        if (name == "None") return {};
        if (name == "True" || name == "False") return boolean(name == "True");
        const auto found = scope_.find(name);
        return found == scope_.end() ? Value{Kind::Symbol, name} : found->second;
    }
    Value sequence(bool tuple) {
        const std::string_view close = tuple ? ")"sv : "]"sv;
        Value result{Kind::Sequence};
        if (take(close)) return result;
        result.elements.push_back(expression());
        if (take(close)) return tuple ? std::move(result.elements.front()) : result;
        expect(",");
        while (!take(close)) {
            result.elements.push_back(expression());
            if (!take(",")) { expect(close); break; }
        }
        return result;
    }
    Value primary() {
        text_ = trim(text_);
        last_delimiter_ = text_.empty() ? 0 : text_.front();
        Value left = atom();
        while (true) {
            if (take(".")) left = compiler_.member(left, identifier());
            else if (take("(")) {
                std::vector<Value> args;
                Scope keywords;
                while (!take(")")) {
                    const std::string_view saved = text_;
                    const size_t name_end = text_.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_");
                    std::string_view name = name_end != text_.npos ? trim(text_.substr(0, name_end)) : ""sv;
                    if (!name.empty()) text_.remove_prefix(name.size());
                    if (!name.empty() && trim(text_).substr(0, 2) != "==" && take("="))
                        require(keywords.emplace(std::string(name), expression()).second, "Duplicate TorchScript keyword argument");
                    else {
                        text_ = saved;
                        args.push_back(expression());
                    }
                    if (!take(",")) { expect(")"); break; }
                }
                left = compiler_.call(std::move(left), std::move(args), keywords, depth_);
            } else if (take("[")) {
                const Value index = expression();
                expect("]");
                if (left.kind == Kind::Sequence) {
                    int at = integer_argument(index);
                    if (at < 0) at += int(left.elements.size());
                    left = Value(left.elements.at(at));
                } else if (left.kind == Kind::Mapping && index.kind == Kind::String) {
                    left = Value(left.fields.at(index.text));
                } else if (left.kind == Kind::Symbol && index.kind == Kind::Symbol) {
                    left.text += '[' + index.text + ']';
                } else throw std::invalid_argument("Unsupported TorchScript indexing expression");
            } else break;
        }
        return left;
    }

    Compiler& compiler_;
    const Scope& scope_;
    std::string_view text_;
    size_t depth_, nesting_ = 0;
    char last_delimiter_ = 0;
};

std::optional<Value> Compiler::block(const Method& source, Scope& scope, size_t& position, size_t end, int indent, size_t depth) {
    require(depth < 128, "TorchScript block nesting exceeds the importer limit");
    LocalFunction parse{[&](std::string_view text) { return Expression(*this, scope, text, depth).parse(); }};
    while (position < end) {
        const Line& line = source.body[position++];
        require(line.indent == indent, "Unexpected TorchScript block indentation");
        const std::string_view code = line.code;
        if (code == "pass") continue;
        if (code == "return") return Value{};
        if (code.substr(0, 7) == "return ") return parse(code.substr(7));
        if (code.substr(0, 3) == "if " && code.back() == ':') {
            const Value condition = parse(code.substr(3, code.size() - 4));
            require(position < end && source.body[position].indent > indent, "Empty TorchScript conditional branch");
            const int child_indent = source.body[position].indent;
            size_t bound = position;
            while (bound < end && source.body[bound].indent > indent) ++bound;
            const bool runtime_check = condition.kind == Kind::Tensor;
            if (runtime_check) {
                const std::string& last = source.body[bound - 1].code;
                require(last.rfind("ops.prim.RaiseException(", 0) == 0 &&
                        (bound == position + 1 || (bound == position + 2 &&
                         source.body[position].code.find(" = torch.format(") != std::string::npos)),
                        "TorchScript runtime branch requires a supported assertion body");
                emit("AssertFalse", {condition});
            }
            const bool selected = runtime_check ? false : truth(condition);
            if (selected) if (std::optional<Value> result = block(source, scope, position, bound, child_indent, depth + 1)) return result;
            position = bound;
            if (position < end && source.body[position].indent == indent && source.body[position].code == "else:") {
                const size_t first = ++position;
                require(first < end && source.body[first].indent > indent, "Empty TorchScript else branch");
                bound = first;
                while (bound < end && source.body[bound].indent > indent) ++bound;
                if (!selected) if (std::optional<Value> result = block(source, scope, position, bound, source.body[first].indent, depth + 1)) return result;
                position = bound;
            }
            continue;
        }
        const size_t equal = code.find(" = ");
        require(equal != code.npos, "Unsupported TorchScript statement " + std::string(code));
        const std::string_view target = trim(code.substr(0, equal));
        const std::string_view name = trim(target.substr(0, target.find(':')));
        require(!name.empty() && name.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_") == name.npos,
                "Unsupported TorchScript assignment target " + std::string(name));
        scope[std::string(name)] = parse(code.substr(equal + 3));
    }
    return {};
}

Value Compiler::invoke(const std::string& key, const std::vector<Value>& args, size_t depth) {
    require(depth < 128, "TorchScript recursive call exceeds the importer limit");
    const Method& source = method(key);
    require(args.size() == source.parameters.size(), "Wrong TorchScript argument count for " + key);
    Scope scope;
    for (size_t i = 0; i < args.size(); ++i) scope.emplace(source.parameters[i], args[i]);
    size_t position = 0;
    const std::optional<Value> result = block(source, scope, position, source.body.size(), source.body.front().indent, depth);
    require(result.has_value(), "TorchScript method has no supported return " + key);
    return *result;
}

Value Compiler::builtin(const std::string& name, const std::vector<Value>& args, const Scope& keywords) {
    if (name == "getattr") {
        require(args.size() == 2 && args[1].kind == Kind::String, "Invalid TorchScript getattr arguments");
        return member(args[0], args[1].text);
    }
    if (name == "annotate" || name == "unchecked_cast") return args.at(1);
    if (name == "ops.prim.RaiseException") {
        const Value message = args.at(0);
        throw std::invalid_argument(message.kind == Kind::String ? message.text : "TorchScript raised an exception");
    }
    if (name == "bool") return boolean(truth(args.at(0)));
    std::string operation = name.rfind("torch.", 0) == 0 ? name.substr(6) : name;
    static const std::set<std::string> mutations{"relu_", "add_", "sub_", "mul_", "div_"};
    const bool inplace = mutations.count(operation) != 0;
    if (inplace) operation.pop_back();
    LocalFunction option{[&](const char* key, size_t position, Value fallback) {
        const auto it = keywords.find(key);
        if (it != keywords.end()) return it->second;
        return position < args.size() ? args[position] : fallback;
    }};
    static const std::map<std::string, std::string> binaries{
        {"add", "Add"}, {"sub", "Sub"}, {"mul", "Mul"}, {"div", "Div"}, {"true_divide", "Div"},
        {"eq", "Eq"}, {"ne", "Ne"}, {"gt", "Gt"}, {"ge", "Ge"}, {"lt", "Lt"}, {"le", "Le"},
    };
    if (const auto it = binaries.find(operation); it != binaries.end()) {
        Value rhs = args.at(1);
        const Value alpha = option("alpha", 2, integer(1));
        if (numeric(alpha) != 1) {
            require(operation == "add" || operation == "sub", "Unsupported TorchScript scalar option");
            rhs = binary("Mul", rhs, alpha);
        }
        const Value result = binary(it->second, args.at(0), rhs);
        return inplace ? functionalize(args[0], result) : result;
    }
    static const std::map<std::string, std::string> unary{
        {"relu", "Relu"}, {"tanh", "Tanh"}, {"sigmoid", "Sigmoid"}, {"silu", "Silu"},
        {"abs", "Abs"}, {"neg", "Neg"}, {"exp", "Exp"}, {"log", "Log"}, {"sqrt", "Sqrt"}, {"square", "Square"},
    };
    if (const auto it = unary.find(operation); it != unary.end()) {
        const Value result = emit(it->second, {args.at(0)});
        return inplace ? functionalize(args[0], result) : result;
    }
    if (operation == "__is__" || operation == "__isnot__") {
        require(args.size() == 2 && (args[0].kind == Kind::None || args[1].kind == Kind::None),
                "TorchScript identity requires a comparison with None");
        const bool both_none = args[0].kind == args[1].kind;
        return boolean(operation == "__is__" ? both_none : !both_none);
    }
    if (operation == "linear") return emit("FullyConnected", args, {{"keep_dims", true}});
    if (operation == "matmul" || operation == "mm" || operation == "bmm") return emit("MatMul", args);
    if (operation == "softmax" || operation == "log_softmax") {
        require(option("dtype", 2, {}).kind == Kind::None, "Unsupported TorchScript softmax dtype override");
        return emit(operation == "softmax" ? "Softmax" : "LogSoftmax", {args.at(0)},
                    {{"axis", integer_argument(option("dim", 1, integer(-1)))}});
    }
    if (operation == "gelu") {
        const Value approximate = option("approximate", 1, {Kind::String, "none"});
        require(approximate.kind == Kind::String && (approximate.text == "none" || approximate.text == "tanh"),
                "Unsupported TorchScript GELU mode");
        return emit(approximate.text == "none" ? "ExactGelu" : "Gelu", {args.at(0)});
    }
    if (operation == "flatten")
        return emit("FlattenRange", {args.at(0)}, {{"start", integer_argument(option("start_dim", 1, integer(0)))},
                                                  {"end", integer_argument(option("end_dim", 2, integer(-1)))}});
    if (operation == "reshape" || operation == "view") return reshape(args.at(0), args.at(1));
    if (operation == "size")
        return args.size() == 1 ? emit("Shape", args) : emit("Size", {args[0]}, {{"axis", integer_argument(args[1])}});
    if (operation == "dim") return emit("Rank", {args.at(0)});
    if (operation == "Int" || operation == "int")
        return args.at(0).kind == Kind::Tensor ? emit("Cast", args, {{"to", dtype_name(DType::Int64)}})
                                               : integer(int64_t(numeric(args.at(0))));
    if (operation == "clone") return emit("Identity", {args.at(0)});
    if (operation == "NumToTensor" || operation == "detach" || operation == "contiguous") return args.at(0);
    if (operation == "dropout") {
        require(!truth(option("train", 2, boolean(true))), "TorchScript training dropout is unsupported in inference");
        return args.at(0);
    }
    if (operation == "permute" || operation == "transpose" || operation == "t") {
        const Value permutation = args.size() > 1 ? args[1] : Value{};
        if (operation == "permute") {
            require(permutation.kind == Kind::Sequence, "Invalid TorchScript permutation");
            Json::Array axes;
            for (const Value& dim : permutation.elements) axes.emplace_back(integer_argument(dim));
            return emit("Transpose", {args[0]}, {{"perm", std::move(axes)}});
        }
        return emit("SwapAxes", {args.at(0)}, {{"a", operation == "t" ? 0 : integer_argument(args.at(1))},
                                             {"b", operation == "t" ? 1 : integer_argument(args.at(2))}});
    }
    if (operation == "unsqueeze" || operation == "squeeze") {
        const Value dim = option("dim", 1, {});
        if (operation == "unsqueeze") return emit("ExpandDims", {args.at(0)}, {{"axis", integer_argument(dim)}});
        return emit("Squeeze", {args.at(0)}, {{"strict", false},
                    {"axes", dim.kind == Kind::None ? Json::Array{} : Json::Array{integer_argument(dim)}}});
    }
    if (operation == "cat" || operation == "stack") {
        require(args.at(0).kind == Kind::Sequence, "TorchScript concat or stack requires a tensor list");
        return emit(operation == "cat" ? "Concat" : "Stack", args[0].elements,
                    {{"axis", integer_argument(option("dim", 1, integer(0)))}});
    }
    if (operation == "adaptive_avg_pool2d") {
        const Value& shape = args.at(1);
        require(shape.kind == Kind::Sequence && shape.elements.size() == 2, "Invalid TorchScript adaptive pooling output size");
        return emit("AdaptiveAvgPool2d", {args.at(0)}, {{"height", integer_argument(shape.elements[0])},
                                                        {"width", integer_argument(shape.elements[1])}});
    }
    if (operation == "_convolution" || operation == "conv2d" || operation == "max_pool2d" || operation == "avg_pool2d") {
        LocalFunction list{[&](size_t at, std::vector<int> fallback) {
            if (at >= args.size() || args[at].kind == Kind::None) return fallback;
            require(args[at].kind == Kind::Sequence, "TorchScript spatial operation requires an integer list");
            std::vector<int> values;
            for (const Value& item : args[at].elements) values.push_back(integer_argument(item));
            return values.empty() ? fallback : values;
        }};
        const bool conv = operation == "_convolution" || operation == "conv2d";
        const std::vector<int> kernel = conv ? std::vector<int>{1, 1} : list(1, {});
        const std::vector<int> step = list(conv ? 3 : 2, kernel), padding = list(conv ? 4 : 3, {0, 0});
        require(step.size() == 2 && padding.size() == 2, "Invalid TorchScript spatial geometry");
        Json::Object attrs{{"strides", Json::Array(step.begin(), step.end())},
                           {"pads", Json::Array{padding[0], padding[1], padding[0], padding[1]}}};
        if (conv) {
            const int group = integer_argument(option("groups", operation == "_convolution" ? 8 : 6, integer(1)));
            if (operation == "_convolution") require(!truth(args.at(6)), "TorchScript transposed convolution is unsupported");
            const std::vector<int> dilation = list(5, {1, 1});
            attrs.emplace("group", group);
            attrs.emplace("dilations", Json::Array(dilation.begin(), dilation.end()));
            return emit("Conv", {args.at(0), args.at(1), option("bias", 2, {})}, std::move(attrs));
        }
        const bool maximum = operation == "max_pool2d";
        attrs.emplace("kernel_shape", Json::Array(kernel.begin(), kernel.end()));
        attrs.emplace("ceil_mode", int(truth(option("ceil_mode", maximum ? 5 : 4, boolean(false)))));
        if (maximum) {
            const std::vector<int> dilation = list(4, {1, 1});
            attrs.emplace("dilations", Json::Array(dilation.begin(), dilation.end()));
        } else {
            require(option("divisor_override", 6, {}).kind == Kind::None, "Unsupported TorchScript pool divisor override");
            attrs.emplace("count_include_pad", int(truth(option("count_include_pad", 5, boolean(true)))));
        }
        return emit(maximum ? "MaxPool" : "AveragePool", {args[0]}, std::move(attrs));
    }
    if (operation == "batch_norm") {
        require(!truth(args.at(5)), "TorchScript training batch normalization is unsupported");
        return emit("AffineBatchNormalization", {args[0], args[3], args[4], args[1], args[2]},
                    {{"axis", 1}, {"epsilon", numeric(args.at(7))}});
    }
    throw std::invalid_argument("Unsupported TorchScript operation " + name);
}

}  // namespace

InferenceSession load_torchscript_session(const std::filesystem::path& path, Backend backend, int device) {
    DeviceScope owner(backend, device);
    const io::TorchScriptArchive archive(path, backend);
    GraphBuilder graph(backend, device);
    Compiler compiler(graph, archive);
    const std::vector<std::string>& parameters = compiler.forward().parameters;
    require(parameters.size() > 1, "TorchScript forward must have at least one tensor input");
    const TensorSpec dynamic{DType::Float32, {}, true, true};
    std::vector<Value> inputs;
    for (size_t i = 1; i < parameters.size(); ++i) inputs.push_back({Kind::Tensor, graph.input(parameters[i], dynamic)});
    const Value result = compiler.run(std::move(inputs));
    LocalFunction add{[&](const std::string& name, const Value& tensor) {
        require(tensor.kind == Kind::Tensor, "TorchScript graph outputs must be tensors");
        graph.output(name, compiler.current(tensor), dynamic);
    }};
    if (result.kind == Kind::Mapping) {
        for (const std::pair<const std::string, Value>& entry : result.fields) {
            const std::string& name = entry.first;
            const Value& value = entry.second;
            add(name, value);
        }
    } else if (result.kind == Kind::Sequence) {
        for (size_t i = 0; i < result.elements.size(); ++i) add("output_" + std::to_string(i), result.elements[i]);
    } else add("output", result);
    return std::move(graph).finish();
}

}  // namespace elman
