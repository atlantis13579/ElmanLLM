#include "Fusion/IR.h"
#include <algorithm>
#include <map>
#include <sstream>
#include <stdexcept>
#include <cstring>

namespace elman::fusion {
namespace {
std::string index(const Dimensions& source, const Dimensions& domain, const std::string& flat) {
    if (source == domain) return flat;
    if (elements(source) == 1) return "0";
    int source_stride = 1, target_stride = 1;
    std::string result = "0";
    for (int d = int(domain.size()) - 1; d >= 0; --d) {
        const int a = d - int(domain.size()) + int(source.size());
        if (a >= 0) {
            if (source[a] != 1) result += "+((" + flat + ")/" + std::to_string(target_stride) + "%" + std::to_string(domain[d]) + ")*" + std::to_string(source_stride);
            source_stride *= source[a];
        }
        target_stride *= domain[d];
    }
    return result;
}
}
std::string generate(const Graph& graph, const Schedule& schedule) {
    std::ostringstream out;
    out << "// Generated from primitive dataflow. FP32, no reassociation or fast math.\n";
    out << R"CUDA(
__device__ float unit_random(unsigned index, unsigned seed, unsigned offset) {
    unsigned v = index + offset * 747796405u + seed * 2891336453u + 1u;
    v ^= v >> 16; v *= 2246822519u; v ^= v >> 13; v *= 3266489917u; v ^= v >> 16;
    return __fmul_rn(__fadd_rn(float(v & 16777215u), .5f), 1.f / 16777216.f);
}
struct Moments { float mean, m2, count; };
__device__ Moments merge_moments(Moments a, Moments b) {
    if (b.count == 0.f) return a;
    if (a.count == 0.f) return b;
    float count = __fadd_rn(a.count,b.count), delta = __fsub_rn(b.mean,a.mean);
    float ratio = __fdiv_rn(b.count,count);
    return {__fmaf_rn(delta,ratio,a.mean),
        __fadd_rn(__fadd_rn(a.m2,b.m2),__fmul_rn(__fmul_rn(delta,delta),__fmul_rn(a.count,ratio))),count};
}
__device__ bool choose_max(float a, int ai, float b, int bi) {
    if (isnan(a) || isnan(b)) return isnan(b) && (!isnan(a) || bi < ai);
    return b > a || (b == a && bi > ai);
}
)CUDA";
    out << "struct Pointers { float* v[" << graph.bindings.size() << "]; };\n";
    out << "struct Scalars { unsigned v[" << std::max<size_t>(1,graph.parameters.size()) << "]; };\n";
    out << "extern \"C\" __global__ void fused(Pointers p, Scalars s) {\n";
    // Region selection rejects every write/read or write/write allocation alias.
    // Read-only inputs may overlap; restrict does not prohibit that case.
    for (size_t binding = 0; binding < graph.bindings.size(); ++binding)
        out << "float* __restrict__ b" << binding << "=p.v[" << binding << "];\n";
    if (schedule.reduction) {
        out << "const int row = int(blockIdx.x)*4+int(threadIdx.y), lane=int(threadIdx.x);\n";
        out << "const bool row_valid = row < " << schedule.rows << ";\n";
    } else {
        out << "#pragma unroll\nfor(int element=0;element<" << schedule.items << ";++element){\n";
        out << "const int i=int(blockIdx.x)*" << 256 * schedule.items
            << "+int(threadIdx.x)+element*256; if(i >= " << schedule.rows << ") continue;\n";
    }
    Dimensions row_shape = schedule.domain;
    if (!row_shape.empty()) row_shape.back() = 1;
    std::vector<bool> scalar;
    std::map<int,std::string> moments;
    auto name = [](int id) { return "v" + std::to_string(id); };
    auto read = [&](int id) { return name(id) + (scalar.at(id) ? "" : "[j]"); };
    auto loop = [&]() {
        out << "#pragma unroll\nfor(int j=0;j<" << schedule.items << ";++j){\n";
        out << "const int column=" << (schedule.vectorized_lanes ? "lane*4+(j/4)*128+j%4" : "lane+j*32")
            << ", i=row*" << schedule.width << "+column;\n";
    };
    for (size_t id = 0; id < graph.values.size(); ++id) {
        const Value& v = graph.values[id]; const std::string target = name(int(id));
        const bool uniform = v.op == Op::Constant || v.op == Op::Parameter ||
            (schedule.reduction && (is_reduction(v.op) || (v.shape == row_shape && row_shape != schedule.domain) || v.shape.empty()));
        scalar.push_back(!schedule.reduction || uniform);
        if (is_reduction(v.op)) {
            if (v.op == Op::MomentsMean || v.op == Op::MomentsVariance) {
                const int source = v.inputs[0];
                if (!moments.count(source)) {
                    const std::string m = "m" + std::to_string(source); moments[source] = m;
                    out << "Moments " << m << "={0.f,0.f,0.f};\n";
                    loop();
                    out << "if(row_valid && column<" << schedule.width << ") " << m << "=merge_moments(" << m << ",{"
                        << read(source) << ",0.f,1.f});\n}\n";
                    out << "for(int offset=16;offset;offset/=2){ Moments other={__shfl_down_sync(0xffffffff," << m << ".mean,offset),"
                        << "__shfl_down_sync(0xffffffff," << m << ".m2,offset),__shfl_down_sync(0xffffffff," << m << ".count,offset)};"
                        << " if(lane+offset<32) " << m << "=merge_moments(" << m << ",other); }\n";
                    out << m << ".mean=__shfl_sync(0xffffffff," << m << ".mean,0);\n";
                    out << m << ".m2=__shfl_sync(0xffffffff," << m << ".m2,0);\n";
                }
                out << "const float " << target << "=" << moments[source] << (v.op == Op::MomentsMean ? ".mean" : ".m2/"+std::to_string(schedule.width)+".f") << ";\n";
                continue;
            }
            const bool maximum = v.op == Op::Maximum;
            if(!maximum && v.slot==int(ReductionOrder::Vector4Warp32)) {
                out << "float a" << id << "[4]={0.f,0.f,0.f,0.f};\n";
                loop();
                out << "if(row_valid && column<" << schedule.width << ") a" << id << "[j%4]=__fadd_rn(a" << id << "[j%4]," << read(v.inputs[0]) << ");\n}\n";
                out << "float " << target << "=__fadd_rn(__fadd_rn(__fadd_rn(a" << id << "[0],a" << id << "[1]),a" << id << "[2]),a" << id << "[3]);\n";
                out << "for(int offset=1;offset<32;offset*=2) " << target << "=__fadd_rn(" << target << ",__shfl_down_sync(0xffffffff," << target << ",offset));\n";
                out << target << "=__shfl_sync(0xffffffff," << target << ",0);\n";
                if(v.op==Op::Mean) {
                    const float factor=1.f/float(schedule.width); uint32_t bits; std::memcpy(&bits,&factor,4);
                    out << target << "=__fmul_rn(" << target << ",__uint_as_float(" << bits << "u));\n";
                }
                continue;
            }
            if (!maximum && v.slot == int(ReductionOrder::Sequential)) {
                out << "float " << target << "=0.f;\n";
                // Every lane consumes the same logical sequence. Shuffle each
                // source slot uniformly before selecting its owner lane.
                for(int j=0;j<schedule.items;++j) for(int lane=0;lane<32 && j*32+lane<schedule.width;++lane)
                    out << target << "=__fadd_rn(" << target << ",__shfl_sync(0xffffffff," << name(v.inputs[0])
                        << (scalar[v.inputs[0]] ? "" : "["+std::to_string(j)+"]") << "," << lane << "));\n";
                if(v.op==Op::Mean) out << target << "=__fdiv_rn(" << target << "," << schedule.width << ".f);\n";
                continue;
            }
            if (!maximum && v.slot == int(ReductionOrder::Block128)) {
                out << "float a" << id << "[4]={0.f,0.f,0.f,0.f};\n";
                loop();
                out << "if(row_valid && column<" << schedule.width << ") a" << id << "[j%4]=__fadd_rn(a" << id << "[j%4]," << read(v.inputs[0]) << ");\n}\n";
                out << "float " << target << "=__fadd_rn(__fadd_rn(a" << id << "[0],a" << id << "[2]),__fadd_rn(a" << id << "[1],a" << id << "[3]));\n";
                out << "for(int offset=16;offset;offset/=2) " << target << "=__fadd_rn(" << target << ",__shfl_down_sync(0xffffffff," << target << ",offset));\n";
                out << target << "=__shfl_sync(0xffffffff," << target << ",0);\n";
                continue;
            }
            out << "float " << target << "=" << (maximum ? "-__int_as_float(0x7f800000)" : "0.f") << ";\n";
            if (maximum) out << "int pos" << id << "=-1;\n";
            loop();
            out << "if(row_valid && column<" << schedule.width << "){ float a=" << read(v.inputs[0]) << ";\n";
            if (maximum) out << "if(choose_max(" << target << ",pos" << id << ",a,column)){" << target << "=a;pos" << id << "=column;}\n";
            else if(v.op==Op::Dot) out << target << "=__fmaf_rn(a," << read(v.inputs[1]) << "," << target << ");\n";
            else out << target << "=__fadd_rn(" << target << ",a);\n";
            out << "}}\nfor(int offset=16;offset;offset/=2){ float a=__shfl_xor_sync(0xffffffff," << target << ",offset);\n";
            if (maximum) out << "int k=__shfl_xor_sync(0xffffffff,pos" << id << ",offset); if(choose_max(" << target << ",pos" << id << ",a,k)){" << target << "=a;pos" << id << "=k;}\n";
            else out << target << "=__fadd_rn(" << target << ",a);\n";
            out << "}\n";
            if (v.op == Op::Mean) out << target << "=__fdiv_rn(" << target << "," << schedule.width << ".f);\n";
            continue;
        }
        out << "float " << target << (scalar.back() ? "" : "["+std::to_string(schedule.items)+"]") << ";\n";
        if (!scalar.back()) loop();
        auto a = [&](int n) { return read(v.inputs.at(n)); };
        std::string expression;
        switch (v.op) {
        case Op::Input: {
            const std::string flat = scalar.back() && schedule.reduction ? "row*"+std::to_string(schedule.width) : "i";
            expression = "b"+std::to_string(v.slot)+"["+index(v.shape,schedule.domain,flat)+"]";
            if (schedule.reduction) expression = "("+std::string(scalar.back() ? "row_valid" : "row_valid && column<"+std::to_string(schedule.width))+" ? "+expression+" : 0.f)";
            break;
        }
        case Op::Constant: expression="__uint_as_float("+std::to_string(v.bits)+"u)"; break;
        case Op::Parameter: expression="__uint_as_float(s.v["+std::to_string(v.slot)+"])"; break;
        case Op::Add: expression="__fadd_rn("+a(0)+","+a(1)+")"; break;
        case Op::Subtract: expression="__fsub_rn("+a(0)+","+a(1)+")"; break;
        case Op::Multiply: expression="__fmul_rn("+a(0)+","+a(1)+")"; break;
        case Op::Divide: expression="__fdiv_rn("+a(0)+","+a(1)+")"; break;
        case Op::Power: expression="powf("+a(0)+","+a(1)+")"; break;
        case Op::Fma: expression="__fmaf_rn("+a(0)+","+a(1)+","+a(2)+")"; break;
        case Op::Negate: expression="(-"+a(0)+")"; break;
        case Op::Exp: expression="expf("+a(0)+")"; break;
        case Op::Tanh: expression="tanhf("+a(0)+")"; break;
        case Op::Erf: expression="erff("+a(0)+")"; break;
        case Op::Sqrt: expression="sqrtf("+a(0)+")"; break;
        case Op::Rsqrt: expression="rsqrtf("+a(0)+")"; break;
        case Op::GreaterEqual: expression="("+a(0)+">="+a(1)+"?1.f:0.f)"; break;
        case Op::Select: expression="("+a(0)+"!=0.f?"+a(1)+":"+a(2)+")"; break;
        case Op::Uniform: expression="unit_random(unsigned(i),__float_as_uint("+a(0)+"),__float_as_uint("+a(1)+"))"; break;
        default: throw std::logic_error("Unlowered fusion primitive");
        }
        out << read(int(id)) << "=" << expression << ";\n";
        if (!scalar.back()) out << "}\n";
    }
    // The runtime retains every externally owned output, including tensor views.
    for (const Store& s : graph.stores) {
        if (!schedule.reduction) out << "b" << s.binding << "[i]=" << name(s.value) << ";\n";
        else if (scalar[s.value]) out << "if(row_valid && lane==0) b" << s.binding << "[row]=" << name(s.value) << ";\n";
        else {
            loop(); out << "if(row_valid && column<" << schedule.width << ") b" << s.binding << "[i]=" << read(s.value) << ";\n}\n";
        }
    }
    if (!schedule.reduction) out << "}\n";
    out << "}\n";
    return out.str();
}
} // namespace elman::fusion
