#include "Fusion/Fusion.h"
#include "Core/Graph.h"
#include "Core/Execution.h"
#include "Core/TensorDispatch.h"
#include "Core/CommandGraph.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

using namespace elman;
namespace {
std::filesystem::path cache;
void check(bool valid,const std::string& message) { if(!valid) throw std::runtime_error(message); }
void near(const std::vector<float>& a,const std::vector<float>& b,const std::string& label,float atol=3e-5f,float rtol=3e-5f) {
    check(a.size()==b.size(),label+" size");
    for(size_t i=0;i<a.size();++i) {
        const bool valid=std::isnan(a[i]) ? std::isnan(b[i]) : std::isinf(a[i]) ? a[i]==b[i] :
            std::isfinite(b[i]) && std::abs(double(a[i])-b[i])<=atol+rtol*std::abs(double(a[i]));
        if(!valid) throw std::runtime_error(label+" at "+std::to_string(i)+": "+std::to_string(a[i])+" versus "+std::to_string(b[i]));
    }
}
fusion::Options options(bool enabled) { fusion::Options o; o.enabled=enabled; o.cache_directory=cache; return o; }
std::vector<float> values(int count,int seed=17) {
    std::mt19937 random(seed); std::uniform_real_distribution<float> distribution(-.8f,.8f);
    std::vector<float> result(count); for(float& x:result) x=distribution(random); return result;
}
struct Result { std::vector<std::vector<float>> outputs,gradients; };
Result expression(bool enabled,int width,int seed) {
    fusion::Policy policy(options(enabled));
    auto x=Tensor::from_data({3,width},values(3*width,seed),Backend::CUDA,true);
    auto bias=Tensor::from_data({width},values(width,seed+1),Backend::CUDA,true);
    fusion::Scope scope;
    std::vector<Tensor> v{x,add(x,bias)};
    std::mt19937 random(seed);
    for(int k=0;k<12;++k) {
        const auto& a=v[random()%v.size()];
        const auto& b=v[random()%v.size()];
        switch(random()%5) {
        case 0: v.push_back(add(mul(a,.3),mul(b,.2))); break;
        case 1: v.push_back(sub(a,mul(b,.15))); break;
        case 2: v.push_back(mul(a,b)); break;
        case 3: v.push_back(div(a,add(mul(b,b),1.))); break;
        default: v.push_back(gelu(a)); break;
        }
    }
    auto centered=sub(v.back(),mean(v.back(),-1,true));
    auto normalized=mul(centered,rsqrt(add(mean(mul(centered,centered),-1,true),.01)));
    auto y=add(normalized,v[1]);
    auto loss=sum_all(mul(y,y));
    loss.backward();
    scope.finish();
    Result result;
    for(const auto& t:v) result.outputs.push_back(t.data());
    result.outputs.push_back(y.data()); result.outputs.push_back(loss.data());
    result.gradients={x.grad(),bias.grad()}; return result;
}
void compare(const Result& a,const Result& b,const std::string& label,float tolerance=8e-5f) {
    check(a.outputs.size()==b.outputs.size() && a.gradients.size()==b.gradients.size(),label+" count");
    for(size_t i=0;i<a.outputs.size();++i) near(a.outputs[i],b.outputs[i],label+" output "+std::to_string(i),tolerance,tolerance);
    for(size_t i=0;i<a.gradients.size();++i) near(a.gradients[i],b.gradients[i],label+" gradient "+std::to_string(i),tolerance,tolerance);
}
Result attention(bool enabled,int width,uint32_t seed,double probability,bool trainable_mask) {
    fusion::Policy policy(options(enabled));
    auto x=Tensor::from_data({2,3,5,width},values(30*width),Backend::CUDA,true);
    auto mask=Tensor::from_data({1,1,5,width},values(5*width),Backend::CUDA,trainable_mask);
    auto upstream=Tensor::from_data(x.shape(),values(30*width,31),Backend::CUDA);
    fusion::Scope scope;
    auto scaled=mul(x,.125);
    auto probability_tensor=attention_softmax(scaled,mask);
    auto output=dropout(probability_tensor,probability,seed,true);
    output.backward(upstream);
    scope.finish();
    return {{scaled.data(),probability_tensor.data(),output.data()},{x.grad(),mask.grad()}};
}
Result low_precision_attention(bool enabled,DType dtype,bool verify) {
    auto settings=options(enabled); settings.verify=verify;
    fusion::Policy policy(settings);
    auto scores=Tensor::from_data({2,3,17},values(102),Backend::CUDA,true,dtype);
    auto mask=Tensor::from_data({1,1,17},values(17,23),Backend::CUDA,true,dtype);
    auto upstream=Tensor::from_data(scores.shape(),values(102,31),Backend::CUDA,false,dtype);
    auto q=Tensor::from_data({1,4,3,4},values(48,37),Backend::CUDA,true,dtype);
    auto k=Tensor::from_data({1,2,5,4},values(40,41),Backend::CUDA,true,dtype);
    auto v=Tensor::from_data({1,2,5,3},values(30,43),Backend::CUDA,true,dtype);
    auto tiled_mask=Tensor::from_data({1,1,3,5},values(15,47),Backend::CUDA,true,dtype);
    auto tiled_upstream=Tensor::from_data({1,4,3,3},values(36,53),Backend::CUDA,false,dtype);
    // These FP8/FP4 operations orchestrate private FP32 attention dispatches.
    // Nonuniform seeds exercise real derivatives and grouped/broadcast sums;
    // disabling verification catches incorrect lowering instead of relying on
    // the runtime's reference comparison to reject a generated kernel.
    fusion::Scope scope;
    auto probability=attention_softmax(scores,mask,2);
    probability.backward(upstream);
    auto output=tiled_attention(q,k,v,tiled_mask,2,2,true,1);
    output.backward(tiled_upstream);
    scope.finish();
    return {{probability.data(),output.data()},
            {scores.grad(),mask.grad(),q.grad(),k.grad(),v.grad(),tiled_mask.grad()}};
}
Result normalization(bool enabled,int width,bool extreme) {
    fusion::Policy policy(options(enabled));
    auto data=values(7*width);
    if(extreme) for(float& x:data) x=1e20f;
    auto x=Tensor::from_data({7,width},data,Backend::CUDA,true);
    auto w=Tensor::from_data({width},values(width,32),Backend::CUDA,true);
    auto b=Tensor::from_data({width},values(width,34),Backend::CUDA,true);
    auto dy=Tensor::from_data({7,width},values(7*width,35),Backend::CUDA);
    fusion::Scope scope;
    auto y=layer_norm(x,w,b,1e-5);
    y.backward(dy); scope.finish();
    const auto output=y.data();
    if(extreme) {
        near(b.data(),std::vector<float>(output.begin(),output.begin()+width),"Constant normalization has zero centered activation",0,0);
        for(float value:output) check(std::isfinite(value),"Extreme constant normalization stays finite");
        for(float value:x.grad()) check(std::isfinite(value),"Extreme constant normalization gradient stays finite");
    }
    return {{output},{x.grad(),w.grad(),b.grad()}};
}
void analytic_gradient() {
    fusion::Policy policy(options(true)); fusion::Scope scope;
    const std::vector<float> xv{-.6f,.2f,.7f,.4f,-.3f,.8f}, bv{.5f,-.4f,.9f};
    auto x=Tensor::from_data({2,3},xv,Backend::CUDA,true), b=Tensor::from_data({3},bv,Backend::CUDA,true);
    auto z=add(mul(x,b),.25); sum_all(mul(z,z)).backward(); scope.finish();
    std::vector<float> dx(6),db(3,0);
    for(int i=0;i<6;++i) { const float g=2.f*(xv[i]*bv[i%3]+.25f); dx[i]=g*bv[i%3]; db[i%3]+=g*xv[i]; }
    near(dx,x.grad(),"Analytic input gradient",1e-6f,1e-6f); near(db,b.grad(),"Analytic broadcast gradient",1e-6f,1e-6f);
}
void compiler_failure() {
    const auto file=cache/"not-a-directory";
    { std::ofstream out(file); out<<"fixture"; }
    auto settings=options(true); settings.cache_directory=file;
    const auto before=fusion::statistics().compile_failures;
    fusion::Policy policy(settings);
    auto x=Tensor::from_data({9},values(9),Backend::CUDA);
    fusion::Scope scope; auto y=mul(add(x,1.),2.); scope.finish();
    auto reference=x.data(); for(float& value:reference) value=(value+1.f)*2.f;
    near(reference,y.data(),"Compiler/cache failure preserves ordinary execution",0,0);
    check(fusion::statistics().compile_failures>before,"Unavailable compilation is reported");
}
void boundaries() {
    fusion::Policy policy(options(true));
    auto x=Tensor::from_data({2,3,17},values(102),Backend::CUDA);
    auto b=Tensor::from_data({1,3,1},values(3),Backend::CUDA);
    std::vector<float> expected;
    { fusion::Policy eager(options(false)); expected=mul(add(x,b),.25).data(); }
    fusion::Scope scope;
    auto first=add(x,b), second=mul(first,.25);
    near(expected,second.data(),"Host read materializes pending graph",0,0);
    auto third=mul(add(x,b),.25);
    auto view=contiguous_view(third,{17},17);
    near(std::vector<float>(expected.begin()+17,expected.begin()+34),view.data(),"Offset view boundary",0,0);
    auto fourth=mul(add(x,b),.25);
    auto reshaped=reshape(fourth,{6,17});
    auto fifth=add(reshaped,1.);
    auto changed=values(102,43); x.assign(changed);
    auto now=mul(add(x,b),.25);
    scope.finish();
    auto expected_fifth=expected; for(float& v:expected_fifth) v+=1;
    near(expected_fifth,fifth.data(),"Reshape and mutation boundary",0,0);
    { fusion::Policy eager(options(false)); near(mul(add(x,b),.25).data(),now.data(),"Mutation visible",0,0); }
    auto into=add(x,b);
    { fusion::Scope transaction; auto y=mul(into,.5); dispatch_tensor_into("add",into,{into,y}); transaction.finish(); }
    { fusion::Policy eager(options(false)); auto base=add(x,b); near(add(base,mul(base,.5)).data(),into.data(),"In-place dispatch boundary",0,0); }
    bool rejected=false;
    try { fusion::Scope pending; auto z=mul(add(x,b),2.); auto invalid=add(z,Tensor::ones({9},Backend::CUDA)); }
    catch(const std::invalid_argument&) { rejected=true; }
    check(rejected,"Invalid broadcast remains an error");
    auto half=Tensor::ones({17},Backend::CUDA,false,DType::Float16);
    { fusion::Scope pending; near(add(mul(half,2.),1.).data(),std::vector<float>(17,3.f),"Unsupported dtype fallback"); pending.finish(); }
    auto zero=Tensor::zeros({0,17},Backend::CUDA);
    { fusion::Scope pending; check(add(zero,zero).data().empty(),"Empty tensors"); pending.finish(); }
}
void streams() {
    fusion::Policy policy(options(true));
    ExecutionStream a(Backend::CUDA),b(Backend::CUDA);
    auto x=Tensor::from_data({4096},values(4096),Backend::CUDA);
    Tensor y,z;
    fusion::Scope scope;
    { StreamScope stream(a,true); y=mul(add(x,1.),2.); }
    { StreamScope stream(b,true); z=sub(mul(y,3.),4.); auto event=b.record(); event.synchronize(); }
    scope.finish();
    fusion::Policy eager(options(false)); near(sub(mul(mul(add(x,1.),2.),3.),4.).data(),z.data(),"Cross-stream dependency",0,0);
}
void stream_lifetime() {
    struct Unwind {};
    fusion::Policy policy(options(true));
    for(bool asynchronous:{false,true}) for(bool unwind:{false,true}) {
        std::weak_ptr<detail::StreamState> recorded;
        Tensor output;
        {
            ExecutionStream stream(Backend::CUDA);
            try {
                StreamScope binding(stream,asynchronous);
                recorded=detail::stream_binding(Backend::CUDA).stream;
                fusion::Scope scope;
                auto input=Tensor::ones({37},Backend::CUDA);
                output=mul(add(input,1.),2.);
                if(unwind) throw Unwind{};
                scope.finish();
            } catch(const Unwind&) {}
            stream.synchronize();
        }
        // Draining a region must release its stream even on fallback unwinding.
        // Vendor workspaces belong to that stream and otherwise outlive users.
        check(recorded.expired(),"Drained fusion region retained its execution stream");
        near(std::vector<float>(37,4.f),output.data(),"Result survives released fusion stream",0,0);
    }
}
void captured_graph() {
    fusion::Policy policy(options(true)); fusion::Scope scope;
    auto x=Tensor::from_data({37},values(37),Backend::CUDA);
    Tensor y;
    CommandGraph graph;
    graph.record([&] { fusion::Policy nested(options(true)); y=mul(add(x,1.),2.); });
    check(graph.command_count()==2,"Nested policies cannot hide commands from CUDA graph recording");
    graph.compile();
    x.assign(std::vector<float>(37,.25f)); graph.replay(); graph.synchronize();
    scope.finish(); near(std::vector<float>(37,2.5f),y.data(),"CUDA graph capture remains a separate execution mechanism",0,0);
}
void exceptional_values() {
    const float nan=std::numeric_limits<float>::quiet_NaN(),inf=std::numeric_limits<float>::infinity();
    auto x=Tensor::from_data({3,5},{0.f,-0.f,inf,-inf,nan, 1e-38f,-1e-38f,1e20f,-1e20f,1.f, -inf,-inf,-inf,-inf,-inf},Backend::CUDA);
    std::vector<float> y,sm,mx;
    { fusion::Policy p(options(false)); y=add(mul(x,2.),x).data(); sm=softmax(x,-1).data(); mx=max(x,-1,true).data(); }
    { fusion::Policy p(options(true)); fusion::Scope s; auto a=add(mul(x,2.),x),b=softmax(x,-1),c=max(x,-1,true); s.finish();
      near(y,a.data(),"Exceptional pointwise",0,0); near(sm,b.data(),"Exceptional softmax"); near(mx,c.data(),"Exceptional reduction",0,0); }
}
void ordered_reductions() {
    auto run=[](bool enabled,const std::vector<float>& data) {
        fusion::Policy policy(options(enabled));
        auto x=Tensor::from_data({1,int(data.size())},data,Backend::CUDA);
        fusion::Scope scope; auto y=mul(sum(add(x,0.),-1,true),2.); scope.finish(); return y.data();
    };
    for(int width:{127,256,512}) {
        const std::vector<float> ordinary(width,1.f);
        near(run(false,ordinary),run(true,ordinary),"Ordinary reduction before extreme reuse",0,0);
        std::vector<float> extreme(width,0.f);
        extreme[0]=3e38f; extreme[32]=3e38f; extreme[64]=-3e38f; extreme[96]=-3e38f;
        near(run(false,extreme),run(true,extreme),"Reduction traversal preserves overflow behavior",0,0);
    }
}
void pointwise_tiles() {
    auto run=[](bool enabled,int rows,int width) {
        fusion::Policy policy(options(enabled));
        auto x=Tensor::from_data({rows,width},values(rows*width),Backend::CUDA,true);
        auto b=Tensor::from_data({1,width},values(width,29),Backend::CUDA,true);
        auto d=Tensor::from_data({rows,1},values(rows,31),Backend::CUDA,true);
        auto dy=Tensor::from_data({rows,width},values(rows*width,37),Backend::CUDA);
        fusion::Scope scope;
        auto a=add(x,b), y=dropout(gelu(mul(a,d)),.2,47,true);
        auto z=add(mul(y,.25),a);
        z.backward(dy); scope.finish();
        return Result{{a.data(),y.data(),z.data()},{x.grad(),b.grad(),d.grad()}};
    };
    for(auto size:std::vector<std::pair<int,int>>{{1,4095},{1,4096},{1,4097},{257,17},{31,1025}})
        compare(run(false,size.first,size.second),run(true,size.first,size.second),"Pointwise tile tail and broadcast");
    auto storage=Tensor::from_data({4100},values(4100),Backend::CUDA);
    auto x=contiguous_view(storage,{4097},0), y=contiguous_view(storage,{4097},1);
    std::vector<float> expected;
    { fusion::Policy policy(options(false)); expected=mul(add(x,y),sub(x,y)).data(); }
    { fusion::Policy policy(options(true)); fusion::Scope scope;
      auto z=mul(add(x,y),sub(x,y)); scope.finish();
      near(expected,z.data(),"Overlapping read-only views with independent output",0,0); }
}
void liveness() {
    fusion::Policy policy(options(true));
    auto x=Tensor::from_data({37},values(37),Backend::CUDA);
    const auto before=fusion::statistics().elided_stores;
    Tensor y;
    { fusion::Scope scope; y=mul(add(x,1.),2.); scope.finish(); }
    check(fusion::statistics().elided_stores>before,"Dead intermediates do not require global stores");
    auto expected=x.data(); for(float& v:expected) v=(v+1)*2;
    near(expected,y.data(),"Live final output after dead store elimination",0,0);
    Tensor alias;
    { fusion::Scope scope; Tensor first=add(x,1.); alias=contiguous_view(first,{17},1);
      y=mul(first,2.); first={}; scope.finish(); }
    auto input=x.data(); std::vector<float> values(input.begin()+1,input.begin()+18); for(float& v:values) v+=1;
    near(values,alias.data(),"A surviving view retains an otherwise dead root store",0,0);
}
void vector_means() {
    auto run=[](bool enabled,int width,bool extreme) {
        fusion::Policy policy(options(enabled));
        auto input=values(17*width);
        if(extreme) for(int row=0;row<17;++row) {
            input[row*width]=3e38f; input[row*width+4]=3e38f;
            input[row*width+8]=-3e38f; input[row*width+12]=-3e38f;
        }
        auto x=Tensor::from_data({17,width},input,Backend::CUDA);
        fusion::Scope scope;
        auto shifted=add(x,.25), average=mean(shifted,-1,true), output=sub(shifted,average);
        scope.finish(); return Result{{shifted.data(),average.data(),output.data()},{}};
    };
    for(int width:{132,384,768,1024}) {
        compare(run(false,width,false),run(true,width,false),"Vector mean ordered graph",0);
        compare(run(false,width,true),run(true,width,true),"Vector mean overflow after cached reuse",0);
    }
}
}
int main(int argc,char** argv) {
    try {
        if(!backend_available(Backend::CUDA)) return 77;
        cache=argc>1 ? argv[1] : "Intermediate/FusionTests";
        fusion::reset_statistics();
        for(int width:{1,17,37,256,777,1024,1025}) for(int seed:{21,42,73})
            compare(expression(false,width,seed),expression(true,width,seed),"Random DAG "+std::to_string(width));
        for(int width:{1,17,256,777,1025}) for(bool mask:{false,true})
            compare(attention(false,width,1337,.2,mask),attention(true,width,1337,.2,mask),"Attention "+std::to_string(width));
        for(double p:{0.,1.,double(.2f),std::nextafter(double(.2f),1.)})
            compare(attention(false,37,47,p,false),attention(true,37,47,p,false),"Dropout boundary");
        for(DType dtype:{DType::Float8E4M3FN,DType::Float8E5M2,DType::Float4E2M1FN})
            for(bool verify:{false,true})
                compare(low_precision_attention(false,dtype,verify),low_precision_attention(true,dtype,verify),
                        std::string(dtype_name(dtype))+" attention fusion, verify="+std::to_string(verify),0);
        const auto compiled=fusion::statistics().compiled;
        compare(attention(false,37,999,std::nextafter(double(.2f),1.),false),attention(true,37,999,std::nextafter(double(.2f),1.),false),"Runtime RNG seed");
        check(fusion::statistics().compiled==compiled,"Changing seed must reuse generated code");
        for(int width:{1,17,64,384,777,1025}) for(bool extreme:{false,true})
            compare(normalization(false,width,extreme),normalization(true,width,extreme),"Layer normalization "+std::to_string(width),1e-4f);
        boundaries(); streams(); stream_lifetime(); captured_graph(); exceptional_values(); ordered_reductions(); analytic_gradient(); pointwise_tiles(); liveness(); vector_means();
        auto stats=fusion::statistics();
        check(stats.launches>0 && stats.verified>0 && stats.fused_commands>stats.launches,"Generated kernels actually execute and fuse dispatches");
        check(stats.compile_failures==0,"Every emitted CUDA program must compile");
        check(stats.rejected==0,"Adversarial generated programs must match the ordinary execution");
        std::cout<<"PASS fusion: compiled="<<stats.compiled<<" cache_hits="<<stats.cache_hits<<" launches="<<stats.launches
                 <<" commands="<<stats.fused_commands<<" verified="<<stats.verified<<" rejected="<<stats.rejected<<'\n';
        compiler_failure();
        return 0;
    } catch(const std::exception& error) { std::cerr<<error.what()<<'\n'; return 1; }
}
