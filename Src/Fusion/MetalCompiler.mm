#include "Fusion/MetalCompiler.h"
#include "Fusion/MetalRuntime.h"
#include "Backends/MPSGraph.h"
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <list>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace elman::fusion {
namespace {
int binding_of(const Graph& graph, const BufferPtr& buffer) {
    if (!buffer) return -1;
    for (size_t i = 0; i < graph.bindings.size(); ++i) {
        const auto& candidate = graph.bindings[i];
        if (&candidate->allocation() == &buffer->allocation() &&
            candidate->raw_pointer() == buffer->raw_pointer() && candidate->count == buffer->count &&
            candidate->dtype == buffer->dtype) return int(i);
    }
    throw std::logic_error("Missing Metal fusion buffer binding");
}

std::string description(const Graph& graph) {
    std::ostringstream out;
    out << "elman.mpsgraph.v1\n";
    for (const auto& buffer : graph.bindings) out << int(buffer->dtype) << ':' << buffer->count << '\n';
    for (const auto& command : graph.commands) {
        out << command.operation << ' ' << binding_of(graph, command.output) << " <-";
        for (const auto& input : command.inputs) out << ' ' << binding_of(graph, input);
        out << "\nparams";
        // Include every attribute: MPSGraph embeds scalar constants in the graph.
        for (int value : command.parameters.v) out << ' ' << value;
        out << '\n';
    }
    out << "targets";
    for (const auto& store : graph.stores) out << ' ' << binding_of(graph, graph.bindings[store.binding]);
    return out.str();
}

std::string fingerprint(const std::string& text) {
    uint64_t hash = 1469598103934665603ull;
    for (unsigned char c : text) hash = (hash ^ c) * 1099511628211ull;
    std::ostringstream out;
    out << std::hex << std::setfill('0') << std::setw(16) << hash;
    return out.str();
}

struct Program final : Executable {
    MPSGraphExecutable* executable = nil;
    std::vector<int> feeds, targets;

    void build(MPSGraphDevice* device, const Graph& region) {
        MPSGraph* graph = [MPSGraph new];
        NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feed_types = [NSMutableDictionary dictionary];
        NSMutableArray<MPSGraphTensor*>* placeholders = [NSMutableArray array];
        std::vector<int> placeholder_bindings;
        std::vector<metal::Tensor> values(region.bindings.size(), nil);
        for (const auto& command : region.commands) {
            metal::Operands inputs{};
            for (size_t i = 0; i < command.inputs.size(); ++i) {
                const int slot = binding_of(region, command.inputs[i]);
                if (slot < 0) continue;
                if (!values[slot]) {
                    metal::Shape dims = @[@(command.inputs[i]->count)];
                    values[slot] = [graph placeholderWithShape:dims dataType:MPSDataTypeFloat32 name:nil];
                    feed_types[values[slot]] = [[MPSGraphShapedType alloc] initWithShape:dims dataType:MPSDataTypeFloat32];
                    [placeholders addObject:values[slot]];
                    placeholder_bindings.push_back(slot);
                }
                inputs.at(i) = [graph reshapeTensor:values[slot]
                                          withShape:metal::storage_shape(command.parameters, int(i)) name:nil];
            }
            // Reuse the ordinary backend's exact kernal definitions. Region
            // selection is dataflow-based; MPSGraph owns Metal kernel lowering.
            auto result = metal::lower(graph, command.operation, inputs, command.parameters, nil);
            if (!result) throw std::runtime_error("Cannot lower Metal fusion operation " + command.operation);
            if (result.dataType != MPSDataTypeFloat32)
                result = [graph castTensor:result toType:MPSDataTypeFloat32 name:nil];
            const int slot = binding_of(region, command.output);
            if (values[slot]) throw std::logic_error("Metal fusion requires single-assignment outputs");
            values[slot] = [graph reshapeTensor:result withShape:metal::storage_shape(command.parameters) name:nil];
        }
        NSMutableArray<MPSGraphTensor*>* results = [NSMutableArray array];
        std::vector<int> result_bindings;
        for (const auto& store : region.stores) {
            const int slot = binding_of(region, region.bindings[store.binding]);
            if (!values[slot] || [results indexOfObjectIdenticalTo:values[slot]] != NSNotFound)
                throw std::logic_error("Metal fusion requires distinct materialized results");
            [results addObject:values[slot]];
            result_bindings.push_back(slot);
        }
        MPSGraphCompilationDescriptor* options = [MPSGraphCompilationDescriptor new];
        options.waitForCompilationCompletion = YES;
        executable = [graph compileWithDevice:device feeds:feed_types targetTensors:results
                            targetOperations:nil compilationDescriptor:options];
        if (!executable || !executable.feedTensors || !executable.targetTensors)
            throw std::runtime_error("Cannot compile Metal fusion region");
        // Compilation can reorder feeds and targets. Bind by tensor identity,
        // never by dictionary enumeration or the original target order.
        for (MPSGraphTensor* tensor in executable.feedTensors) {
            const NSUInteger index = [placeholders indexOfObjectIdenticalTo:tensor];
            if (index == NSNotFound) throw std::runtime_error("Invalid Metal fusion feed mapping");
            feeds.push_back(placeholder_bindings.at(index));
        }
        for (MPSGraphTensor* tensor in executable.targetTensors) {
            const NSUInteger index = [results indexOfObjectIdenticalTo:tensor];
            if (index == NSNotFound) throw std::runtime_error("Invalid Metal fusion target mapping");
            targets.push_back(result_bindings.at(index));
        }
        if (targets.size() != region.stores.size()) throw std::runtime_error("Missing Metal fusion targets");
    }

    void launch(const Graph& graph, const Schedule&) override {
        @autoreleasepool {
            std::vector<BufferPtr> inputs, outputs;
            for (int slot : feeds) inputs.push_back(graph.bindings.at(slot));
            for (int slot : targets) outputs.push_back(graph.bindings.at(slot));
            encode_metal(executable, inputs, outputs);
        }
    }
};
} // namespace

std::shared_ptr<Executable> compile_metal(const Graph& graph, const Options& options, Statistics& stats) {
    // Thread-local verification state and a bounded LRU keep dynamic shapes from
    // retaining an unbounded number of compiled graphs. Keys contain no buffers.
    using Cache = std::list<std::pair<std::string, std::shared_ptr<Executable>>>;
    thread_local Cache cached;
    thread_local std::unordered_map<std::string, Cache::iterator> index;
    @autoreleasepool {
        MPSGraphDevice* device = metal_graph_device();
        const std::string source = description(graph);
        const std::string key = std::to_string(reinterpret_cast<uintptr_t>((__bridge void*)device)) + '\n' +
                                options.cache_directory.string() + '\n' + source;
        if (const auto found = index.find(key); found != index.end()) {
            cached.splice(cached.begin(), cached, found->second);
            ++stats.cache_hits;
            return found->second->second;
        }
        auto program = std::make_shared<Program>();
        try {
            @try { program->build(device, graph); }
            @catch (NSException* exception) {
                const char* reason = exception.reason.UTF8String;
                throw std::runtime_error(reason ? reason : "Metal fusion compilation exception");
            }
            ++stats.compiled;
        } catch (const std::exception& error) {
            program->rejected = true;
            program->diagnostic = error.what();
            ++stats.compile_failures;
        }
        // Diagnostic graphs are inspectable; executable reuse is in memory.
        // A diagnostic I/O failure must not prevent ordinary tensor execution.
        std::error_code error;
        const auto directory = options.cache_directory / "Metal";
        std::filesystem::create_directories(directory, error);
        if (!error) {
            std::ofstream log(directory / (fingerprint(source) + ".mpsgraph.txt"));
            log << source << '\n' << program->diagnostic << '\n';
        }
        cached.emplace_front(key, program);
        try { index.emplace(key, cached.begin()); }
        catch (...) { cached.pop_front(); throw; }
        if (cached.size() > 128) { index.erase(cached.back().first); cached.pop_back(); }
        return program;
    }
}
} // namespace elman::fusion
