#include "Fusion/Fusion.h"
#include "Fusion/IR.h"
#include "Fusion/Compiler.h"
#include "Core/Device.h"
#include "Core/Execution.h"
#include <cmath>
#include <cstring>
#include <exception>
#include <fstream>
#include <stdexcept>
#include <atomic>
#include <cstdlib>
#include <algorithm>
#include <unordered_map>
#ifdef ELMAN_HAS_CUDA
#include "Fusion/CudaCompiler.h"
#include "Backends/CudaExecution.h"
#include "Core/CommandGraph.h"
namespace elman { void dispatch_cuda(const std::string&, const BufferPtr&, const std::vector<BufferPtr>&, const KernelParams&); }
#endif
#ifdef ELMAN_HAS_METAL
#include "Fusion/MetalCompiler.h"
#include "Backends/Metal.h"
#endif

namespace elman::fusion {
namespace {
std::atomic<bool>& master_switch() {
    static std::atomic<bool> value([] {
        const char* setting=std::getenv("ELMAN_FUSION");
        return !setting || std::string(setting)!="0";
    }());
    return value;
}
thread_local Options policy;
thread_local Statistics counters;
thread_local int depth = 0;
thread_local bool executing = false;
thread_local Graph pending;
thread_local int device = -1;
thread_local Backend backend = Backend::CPU;
thread_local detail::StreamBinding binding;
struct Executing {
    Executing() { executing=true; }
    ~Executing() { executing=false; }
};
bool overlap(const BufferPtr& a,const BufferPtr& b) { return a && b && &a->allocation()==&b->allocation(); }
#if defined(ELMAN_HAS_CUDA) || defined(ELMAN_HAS_METAL)
void original(const Graph& graph) {
    for(const Command& c:graph.commands) {
#ifdef ELMAN_HAS_CUDA
        if(backend==Backend::CUDA) { dispatch_cuda(c.operation,c.output,c.inputs,c.parameters); continue; }
#endif
#ifdef ELMAN_HAS_METAL
        if(backend==Backend::Metal) { dispatch_metal(c.operation,c.output,c.inputs,c.parameters); continue; }
#endif
        throw std::logic_error("Unsupported fusion backend");
    }
}
std::shared_ptr<Executable> compile_region(const Graph& graph,const Schedule& schedule) {
#ifdef ELMAN_HAS_CUDA
    if(backend==Backend::CUDA) return compile(graph,schedule,policy,counters);
#endif
#ifdef ELMAN_HAS_METAL
    if(backend==Backend::Metal) return compile_metal(graph,policy,counters);
#endif
    throw std::logic_error("Unsupported fusion compiler");
}
void submit_region(const Graph& graph,Executable& executable,const Schedule& schedule) {
#ifdef ELMAN_HAS_CUDA
    if(backend==Backend::CUDA) { submit(graph,executable,schedule); return; }
#endif
    executable.launch(graph,schedule);
}
bool close(float a,float b) {
    if(std::isnan(a)||std::isnan(b)) return std::isnan(a)&&std::isnan(b);
    if(std::isinf(a)||std::isinf(b)) return a==b;
    if(a==0 && b==0) return std::signbit(a)==std::signbit(b);
    return std::abs(double(a)-double(b)) <= 2e-5+2e-5*std::abs(double(a));
}
void execute(Graph& graph,bool fallback) {
    if(fallback || !master_switch().load()) { original(graph); return; }
    // A single native Metal operation already uses an MPSGraph executable.
    if(backend==Backend::Metal && graph.commands.size()<2) { original(graph); return; }
    // A root buffer is dead only when every shared owner is inside this region.
    // Tensor aliases/views retain their root owner, so they conservatively keep
    // the store even after the original tensor object has been destroyed.
    std::unordered_map<const Buffer*,long> owned;
    for(const auto& b:graph.bindings) if(b) ++owned[b.get()];
    for(const auto& c:graph.commands) {
        ++owned[c.output.get()];
        for(const auto& b:c.inputs) if(b) ++owned[b.get()];
    }
    graph.stores.erase(std::remove_if(graph.stores.begin(),graph.stores.end(),[&](const Store& s) {
        const auto& b=graph.bindings[s.binding];
        const bool dead=&b->allocation()==b.get() && b.use_count()==owned[b.get()];
        if(dead) ++counters.elided_stores;
        return dead;
    }),graph.stores.end());
    if(graph.stores.empty()) return;
    Schedule schedule;
    if(!analyze(graph,schedule)) { original(graph); return; }
    size_t operations=0;
    for(const auto& v:graph.values) if(v.op!=Op::Input && v.op!=Op::Constant && v.op!=Op::Parameter) ++operations;
    if(operations<2) { original(graph); return; }
    std::shared_ptr<Executable> executable;
    try { executable=compile_region(graph,schedule); }
    catch(const std::exception& error) {
        ++counters.compile_failures;
        std::ofstream log(policy.cache_directory/"runtime-errors.log",std::ios::app); log<<error.what()<<'\n';
        original(graph); return;
    }
    if(executable->rejected) { original(graph); return; }
    if(policy.verify && !executable->verified) {
        original(graph);
        std::vector<std::vector<uint8_t>> reference;
        for(const Store& s:graph.stores) reference.push_back(backend_download_raw(graph.bindings[s.binding]));
        submit_region(graph,*executable,schedule);
        bool valid=true;
        for(size_t k=0;k<graph.stores.size();++k) {
            const auto actual=backend_download_raw(graph.bindings[graph.stores[k].binding]);
            if(actual.size()!=reference[k].size()) { valid=false; continue; }
            for(size_t i=0;i<actual.size();i+=sizeof(float)) {
                float a,b; std::memcpy(&a,reference[k].data()+i,4); std::memcpy(&b,actual.data()+i,4);
                if(!close(a,b)) { valid=false; break; }
            }
        }
        if(!valid) {
            executable->rejected=true; ++counters.rejected;
            for(size_t k=0;k<graph.stores.size();++k) backend_upload_raw(graph.bindings[graph.stores[k].binding],reference[k]);
            std::ofstream log(policy.cache_directory/"verification.log",std::ios::app);
            log<<"Rejected region with "<<graph.commands.size()<<" dispatches and "<<graph.values.size()<<" IR values\n";
            return;
        }
        executable->verified=true; ++counters.verified;
    } else submit_region(graph,*executable,schedule);
    ++counters.launches; counters.fused_commands+=graph.commands.size();
}
#endif
void drain(bool fallback) {
    if(executing || pending.commands.empty()) return;
    Executing guard;
    Graph graph=std::move(pending); pending={};
    // Retain the recorded stream only while submitting this region. Leaving a
    // shared binding on an empty queue keeps the stream (and its BLAS workspace)
    // alive after the caller releases it, including exception fallback paths.
    auto submitted_binding=std::move(binding); binding={};
#if defined(ELMAN_HAS_CUDA) || defined(ELMAN_HAS_METAL)
    DeviceScope owner(backend,device);
    auto& current=detail::stream_binding(backend);
    const auto previous=current; current=submitted_binding;
    try { execute(graph,fallback); }
    catch(...) { current=previous; throw; }
    current=previous;
#endif
}
}
Policy::Policy(const Options& options) : previous_(policy) { flush(); policy=options; }
Policy::~Policy() noexcept(false) {
    try { flush(); } catch(...) { policy=previous_; if(!std::uncaught_exceptions()) throw; return; }
    policy=previous_;
}
bool enabled() { return master_switch().load() && policy.enabled; }
void set_enabled(bool value) { flush(); master_switch().store(value); }
Scope::Scope() : active_(enabled()) { if(active_) ++depth; }
Scope::~Scope() noexcept(false) {
    if(!active_) return;
    active_=false; --depth;
    if(depth) return;
    if(std::uncaught_exceptions()) { try { drain(true); } catch(...) {} }
    else flush();
}
void Scope::finish() { if(active_) { active_=false; if(--depth==0) flush(); } }
Statistics statistics() { return counters; }
void reset_statistics() { flush(); counters={}; }
void flush() { drain(false); }
void access(const BufferPtr& buffer) {
    if(executing || pending.commands.empty()) return;
    for(const auto& b:pending.bindings) if(overlap(b,buffer)) { ++counters.barriers; flush(); return; }
}
bool enqueue(const std::string& op,const BufferPtr& output,const std::vector<BufferPtr>& inputs,const KernelParams& params) {
#if defined(ELMAN_HAS_CUDA) || defined(ELMAN_HAS_METAL)
    if(executing) return false;
    if(!enabled()) { flush(); return false; }
    if(!depth) return false;
    // A callback may install its own policy during command recording. Capture
    // must still see every original dispatch, including already cached regions.
#ifdef ELMAN_HAS_CUDA
    if(output->backend==Backend::CUDA && (detail::recording_cuda_graph() || detail::lowering_cuda_graph())) { flush(); return false; }
#endif
    if(output->backend!=Backend::CUDA && output->backend!=Backend::Metal) { flush(); return false; }
    // Metal's random samples are produced by MPS Philox outside MPSGraph. Keep
    // them on the ordinary path; CUDA's stateless RNG is a different contract.
    if(output->backend==Backend::Metal && op=="dropout") { ++counters.barriers; flush(); return false; }
    if(output->dtype!=DType::Float32 || !has_definition(op)) { ++counters.barriers; flush(); return false; }
    const auto& current=detail::stream_binding(output->backend);
    if(!pending.commands.empty() && (backend!=output->backend || device!=output->device_ordinal() || current.stream!=binding.stream ||
            current.asynchronous!=binding.asynchronous || current.pinned_staging!=binding.pinned_staging)) flush();
    for(const auto& input:inputs) if(overlap(output,input)) { flush(); return false; }
    for(const auto& b:pending.bindings) if(overlap(output,b)) { flush(); break; }
    // An offset/reshaped alias of an earlier output cannot be read from an SSA value.
    for(const auto& input:inputs) if(input) for(const Store& s:pending.stores) {
        const auto& b=pending.bindings[s.binding];
        if(overlap(input,b) && (input->raw_pointer()!=b->raw_pointer() || input->count!=b->count)) { flush(); goto aliases_checked; }
    }
aliases_checked:
    if(pending.commands.size()>=64) flush();
    const Command command{op,output,inputs,params};
    auto attempt=[&]() {
        const auto nv=pending.values.size(), nb=pending.bindings.size(), np=pending.parameters.size();
        const auto ns=pending.stores.size(), nc=pending.commands.size();
        auto rollback=[&]() { pending.values.resize(nv); pending.bindings.resize(nb); pending.parameters.resize(np);
                             pending.stores.resize(ns); pending.commands.resize(nc); };
        try {
            Schedule schedule;
            if(!lower(pending,command) || !analyze(pending,schedule)) { rollback(); return false; }
        } catch(const std::invalid_argument&) { rollback(); return false; }
        catch(...) { rollback(); throw; }
        return true;
    };
    if(!attempt()) { ++counters.barriers; flush(); if(!attempt()) return false; }
    backend=output->backend; device=output->device_ordinal(); binding=current;
    return true;
#else
    return false;
#endif
}
} // namespace elman::fusion
