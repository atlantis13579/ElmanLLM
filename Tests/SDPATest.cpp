#include "Core/Graph.h"
#include "Fusion/Fusion.h"
#include "IO/Json.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace elman;
namespace {
void check(bool value,const char* message) { if(!value) throw std::runtime_error(message); }
std::vector<float> data(int n,int salt) { std::vector<float> x(n); for(int i=0;i<n;++i) x[i]=.4f*std::sin(float(i*13+salt)); return x; }
struct Result { std::vector<float> y,q,k,v,m; };
Result run(Backend backend,bool fused,int length,int width,bool causal,int mask_kind,double drop,bool gqa,bool broadcast) {
    fusion::Options options; options.enabled=fused; options.cache_directory="Intermediate/SdpaFusion"; fusion::Policy policy(options);
    Tensor q=Tensor::from_data({2,4,7,width},data(2*4*7*width,1),backend,true);
    const int kh=gqa?2:4,kb=broadcast?1:2;
    Tensor k=Tensor::from_data({kb,kh,length,width},data(kb*kh*length*width,2),backend,true);
    Tensor v=Tensor::from_data({2,kh,length,5},data(2*kh*length*5,3),backend,true), mask;
    if(mask_kind) {
        auto values=data(7*length,4);
        for(int i=0;i<7*length;++i) {
            if(mask_kind==1) values[i]=(i/length==2 || i%5==0)?0.f:1.f;
            else if(i/length==2 || i%5==0) values[i]=-std::numeric_limits<float>::infinity();
        }
        mask=Tensor::from_data({7,length},values,backend,mask_kind==2,mask_kind==1?DType::Bool:DType::Float32);
    }
    fusion::Scope scope;
    auto y=scaled_dot_product_attention(q,k,v,mask,drop,causal,.37,gqa,123);
    y.backward(Tensor::from_data(y.shape(),data(int(y.numel()),5),backend)); scope.finish();
    return {y.data(),q.grad(),k.grad(),v.grad(),mask.defined()?mask.grad():std::vector<float>{}};
}
void compare(const Result& a,const Result& b) {
    for(auto pair:{std::make_pair(&a.y,&b.y),{&a.q,&b.q},{&a.k,&b.k},{&a.v,&b.v},{&a.m,&b.m}}) {
        check(pair.first->size()==pair.second->size(),"SDPA output/gradient size");
        for(size_t i=0;i<pair.first->size();++i) {
            float x=(*pair.first)[i],y=(*pair.second)[i];
            if(!std::isfinite(x) || !std::isfinite(y) || std::abs(x-y)>3e-5f+3e-5f*std::abs(x))
                throw std::runtime_error("SDPA output/gradient mismatch at "+std::to_string(i)+": "+std::to_string(x)+" / "+std::to_string(y));
        }
    }
}
std::vector<float> numbers(const Json& j) {
    std::vector<float> result;
    for(const auto& x:j.at("values").array()) result.push_back(j.bool_or("boolean",false)?float(x.boolean()):float(x.number()));
    return result;
}
Tensor fixture_tensor(const Json& j,Backend backend,bool grad) {
    if(j.is_null()) return {};
    Shape shape; for(const auto& d:j.at("shape").array()) shape.push_back(d.integer());
    const bool boolean=j.bool_or("boolean",false);
    return Tensor::from_data(shape,numbers(j),backend,grad&&!boolean,boolean?DType::Bool:DType::Float32);
}
void reference_fixtures(const std::string& path,Backend backend) {
    const auto fixtures=Json::read(path);
    for(const auto& c:fixtures.at("cases").array()) {
        auto q=fixture_tensor(c.at("q"),backend,true),k=fixture_tensor(c.at("k"),backend,true),v=fixture_tensor(c.at("v"),backend,true);
        auto mask=fixture_tensor(c.at("mask"),backend,true);
        auto y=scaled_dot_product_attention(q,k,v,mask,0,c.at("causal").boolean(),c.at("scale").number(),c.at("gqa").boolean());
        y.backward(fixture_tensor(c.at("dy"),backend,false));
        Result reference{numbers(c.at("y")),numbers(c.at("dq")),numbers(c.at("dk")),numbers(c.at("dv")),
            c.at("dm").is_null()?std::vector<float>{}:numbers(c.at("dm"))};
        compare(reference,{y.data(),q.grad(),k.grad(),v.grad(),mask.defined()?mask.grad():std::vector<float>{}});
    }
    std::cout<<"PASS independent PyTorch SDPA fixtures: "<<fixtures.at("cases").array().size()<<'\n';
}
}
int main(int argc,char** argv) {
    try {
        Backend backend=argc>1?parse_backend(argv[1]):Backend::CUDA;
        if(!backend_available(backend)) return 77;
        check(fusion::enabled(),"Fusion defaults to enabled");
        for(int length:{1,17,256,1025}) for(int mask:{0,1,2}) {
            compare(run(Backend::CPU,false,length,7,false,mask,0,true,true),run(backend,true,length,7,false,mask,0,true,true));
        }
        for(double drop:{0.,.2,1.}) for(bool causal:{false,true})
            compare(run(backend,false,37,13,causal,0,drop,false,false),run(backend,true,37,13,causal,0,drop,false,false));
        fusion::set_enabled(false); check(!fusion::enabled(),"Global fusion switch disables nested policies");
        compare(run(backend,false,17,7,true,0,0,false,false),run(backend,true,17,7,true,0,0,false,false));
        fusion::set_enabled(true);
        if(argc>2) reference_fixtures(argv[2],backend);
        std::cout<<"PASS SDPA forward, gradients, masks, dropout, GQA, broadcast and global switch\n";
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
