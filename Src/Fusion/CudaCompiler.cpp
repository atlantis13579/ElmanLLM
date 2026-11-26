#include "Fusion/CudaCompiler.h"
#include "Backends/CudaStream.h"
#include <cuda.h>
#include <nvrtc.h>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <sstream>
#include <stdexcept>

namespace elman {
void submit_fusion_cuda(const std::vector<BufferPtr>& inputs, const std::vector<BufferPtr>& outputs,
                        const std::function<void()>& launch);
}
namespace elman::fusion {
namespace {
void driver(CUresult status, const char* action) {
    if (status == CUDA_SUCCESS) return;
    const char* message = nullptr; cuGetErrorString(status,&message);
    throw std::runtime_error(std::string(action)+": "+(message ? message : "CUDA driver error"));
}
void rtc(nvrtcResult status, const char* action) {
    if (status != NVRTC_SUCCESS) throw std::runtime_error(std::string(action)+": "+nvrtcGetErrorString(status));
}
std::string fingerprint(const std::string& text) {
    uint64_t a = 14695981039346656037ull, b = 7809847782465536322ull;
    for (unsigned char c : text) { a = (a ^ c) * 1099511628211ull; b = (b ^ (c+1u)) * 14029467366897019727ull; }
    std::ostringstream s; s << std::hex << std::setfill('0') << std::setw(16) << a << std::setw(16) << b; return s.str();
}
std::string read(const std::filesystem::path& path) {
    std::ifstream in(path,std::ios::binary); if (!in) return {};
    return std::string(std::istreambuf_iterator<char>(in),{});
}
void write(const std::filesystem::path& path, const std::string& text) {
    std::ofstream out(path,std::ios::binary|std::ios::trunc);
    if (!out || !out.write(text.data(),std::streamsize(text.size()))) throw std::runtime_error("Cannot write fusion cache: "+path.string());
}
struct Program {
    nvrtcProgram handle = nullptr;
    ~Program() { if (handle) nvrtcDestroyProgram(&handle); }
};
struct Module final : Executable {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    CUcontext context = nullptr;
    ~Module() override {
        if (module && cuCtxPushCurrent(context) == CUDA_SUCCESS) { cuModuleUnload(module); CUcontext previous; cuCtxPopCurrent(&previous); }
    }
    void launch(const Graph& graph, const Schedule& schedule) override {
        std::vector<void*> pointers;
        for (const auto& buffer : graph.bindings) pointers.push_back(buffer->raw_pointer());
        uint32_t unused = 0;
        void* arguments[]{pointers.data(), graph.parameters.empty() ? &unused : const_cast<uint32_t*>(graph.parameters.data())};
        const unsigned tile = 256u * unsigned(schedule.items);
        const unsigned blocks = schedule.reduction ? (unsigned(schedule.rows)+3)/4 : (unsigned(schedule.rows)+tile-1)/tile;
        driver(cuLaunchKernel(function,blocks,1,1,schedule.reduction ? 32 : 256,schedule.reduction ? 4 : 1,1,0,
                              current_cuda_stream(),arguments,nullptr),"Launch generated fusion kernel");
    }
};
}
static std::shared_ptr<Executable> compile_source(const std::string& source, const Options& options, Statistics& stats) {
    // In-memory entries are per CUDA context and thread; verification decisions cannot race.
    thread_local std::map<std::string,std::shared_ptr<Executable>> cache;
    CUcontext context = nullptr; driver(cuCtxGetCurrent(&context),"Get fusion CUDA context");
    if (!context) throw std::runtime_error("Fusion requires an initialized CUDA context");
    CUdevice device; driver(cuCtxGetDevice(&device),"Get fusion CUDA device");
    int major=0,minor=0,rtc_major=0,rtc_minor=0;
    driver(cuDeviceGetAttribute(&major,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,device),"Get compute capability");
    driver(cuDeviceGetAttribute(&minor,CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,device),"Get compute capability");
    rtc(nvrtcVersion(&rtc_major,&rtc_minor),"Get NVRTC version");
    const std::string architecture="--gpu-architecture=compute_"+std::to_string(major)+std::to_string(minor);
    const std::string contract="elman.fusion.v1\n"+architecture+"\nnvrtc="+std::to_string(rtc_major)+"."+std::to_string(rtc_minor)+
        "\n--std=c++17 --fmad=false --ftz=false --prec-div=true --prec-sqrt=true\n"+source;
    const std::string key=contract+"\ncontext="+std::to_string(reinterpret_cast<uintptr_t>(context))+"\ncache="+options.cache_directory.string();
    if (auto found=cache.find(key);found!=cache.end()) { ++stats.cache_hits; return found->second; }
    auto module=std::make_shared<Module>(); module->context=context;
    std::filesystem::create_directories(options.cache_directory);
    const auto base=options.cache_directory/fingerprint(contract);
    const std::filesystem::path code=base.string()+".cu", meta=base.string()+".key", binary=base.string()+".ptx", checksum=base.string()+".hash";
    // Source equality and artifact hashes reject collisions, incomplete writes, and damaged caches.
    std::string ptx;
    if (read(meta)==contract && read(code)==source) { ptx=read(binary); if (ptx.empty() || read(checksum)!=fingerprint(ptx)) ptx.clear(); }
    if (!ptx.empty()) ++stats.cache_hits;
    else {
        Program program;
        rtc(nvrtcCreateProgram(&program.handle,source.c_str(),"elman_generated.cu",0,nullptr,nullptr),"Create fusion program");
        const char* flags[]{architecture.c_str(),"--std=c++17","--fmad=false","--ftz=false","--prec-div=true","--prec-sqrt=true"};
        const nvrtcResult status=nvrtcCompileProgram(program.handle,6,flags);
        size_t length=0; rtc(nvrtcGetProgramLogSize(program.handle,&length),"Get fusion compiler log size");
        std::string log(length,'\0'); if(length) rtc(nvrtcGetProgramLog(program.handle,log.data()),"Get fusion compiler log");
        write(code,source); write(base.string()+".log",log);
        if(status!=NVRTC_SUCCESS) { module->rejected=true; module->diagnostic=log; ++stats.compile_failures; cache.emplace(key,module); return module; }
        rtc(nvrtcGetPTXSize(program.handle,&length),"Get generated PTX size"); ptx.resize(length);
        rtc(nvrtcGetPTX(program.handle,ptx.data()),"Get generated PTX");
        write(binary,ptx); write(checksum,fingerprint(ptx)); write(meta,contract); ++stats.compiled;
    }
    driver(cuModuleLoadData(&module->module,ptx.c_str()),"Load generated fusion module");
    driver(cuModuleGetFunction(&module->function,module->module,"fused"),"Get generated fusion function");
    cache.emplace(key,module); return module;
}
std::shared_ptr<Executable> compile(const Graph& graph, const Schedule& schedule, const Options& options, Statistics& stats) {
    // Runtime scalar values (including RNG seeds) are deliberately absent from the key.
    // Structural identity is checked in full, rather than trusting a hash collision.
    thread_local std::map<std::string,std::shared_ptr<Executable>> plans;
    CUcontext context=nullptr; driver(cuCtxGetCurrent(&context),"Get fusion plan context");
    std::string key=std::to_string(reinterpret_cast<uintptr_t>(context))+"\n"+options.cache_directory.string()+"\n";
    auto integer=[&](int value) { key.append(reinterpret_cast<const char*>(&value),sizeof(value)); };
    integer(schedule.reduction); integer(schedule.rows); integer(schedule.width); integer(schedule.items);
    integer(schedule.vectorized_lanes);
    integer(int(graph.bindings.size())); integer(int(graph.parameters.size())); integer(int(graph.values.size()));
    for(const Value& v:graph.values) {
        integer(int(v.op)); integer(int(v.bits)); integer(v.slot); integer(int(v.shape.size()));
        for(int d:v.shape) integer(d);
        integer(int(v.inputs.size())); for(int input:v.inputs) integer(input);
    }
    for(const Store& s:graph.stores) { integer(s.value); integer(s.binding); }
    if(auto found=plans.find(key);found!=plans.end()) { ++stats.cache_hits; return found->second; }
    auto executable=compile_source(generate(graph,schedule),options,stats);
    plans.emplace(std::move(key),executable); return executable;
}
void submit(const Graph& graph, Executable& executable, const Schedule& schedule) {
    std::vector<BufferPtr> outputs;
    for (const Store& s : graph.stores) outputs.push_back(graph.bindings[s.binding]);
    submit_fusion_cuda(graph.bindings,outputs,[&] { executable.launch(graph,schedule); });
}
} // namespace elman::fusion
