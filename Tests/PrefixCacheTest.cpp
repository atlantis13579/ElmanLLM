#include "Inference/LLM/PrefixCache.h"
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
#include <thread>

using namespace elman;
namespace {
int checks=0;
void check(bool value,const char* message) { ++checks;if(!value)throw std::runtime_error(message); }
template<class E=std::invalid_argument,class F> void rejects(F action) {
    bool caught=false;try{action();}catch(const E&){caught=true;}check(caught,"Expected prefix-cache rejection");
}
std::vector<int32_t> tokens(int size,int start=0) {
    std::vector<int32_t> result(size);std::iota(result.begin(),result.end(),start);return result;
}
Tensor payload(Backend backend,DType dtype,const std::vector<int32_t>& ids,int begin,int layer,bool key) {
    std::vector<float> data;
    for(size_t i=size_t(begin);i<ids.size();++i)for(int channel=0;channel<2;++channel)
        data.push_back(float(ids[i]%31-15)*0.5f+float(layer+channel+int(key))*0.125f);
    return Tensor::from_data({1,1,int(ids.size())-begin,2},data,backend,false,dtype);
}
void append(PagedKVCache& cache,const std::vector<int32_t>& ids) {
    int begin=cache.length();std::unique_ptr<PagedKVAppend> transaction=cache.append(int(ids.size())-begin);
    for(int layer=0;layer<cache.shape().layers;++layer)transaction->write(layer,payload(cache.backend(),cache.dtype(),ids,begin,layer,true),
        payload(cache.backend(),cache.dtype(),ids,begin,layer,false));
    transaction->commit();
}
void equal(PagedKVCache& cache,const std::vector<int32_t>& ids) {
    check(cache.length()==int(ids.size()),"Prefix length differs");
    for(int layer=0;layer<cache.shape().layers;++layer) {
        PagedKVLayer value=cache.read(layer);
        check(value.key.raw_data()==payload(cache.backend(),cache.dtype(),ids,0,layer,true).raw_data(),"Cached key differs");
        check(value.value.raw_data()==payload(cache.backend(),cache.dtype(),ids,0,layer,false).raw_data(),"Cached value differs");
    }
}
void lifecycle(Backend backend,DType dtype,int page,bool reuse) {
    size_t baseline=memory_stats(backend).live_bytes(),host=memory_stats(Backend::CPU).live_bytes();
    {
        ExecutionStream stream(backend);StreamScope execution(stream);AllocationPool allocations(backend,0,reuse);
        std::shared_ptr<elman::PagedKVPool> pool=std::make_shared<PagedKVPool>(PagedKVShape{2,1,2,page,32},backend,dtype);
        PrefixCache store(pool,4,2);PagedKVCache source(*pool,1,page*8);
        std::vector<int32_t> prompt=tokens(page*4+1);append(source,prompt);size_t live=pool->stats().live;
        PrefixPublication published=store.publish(prompt,source);
        check(published.inserted && published.tokens==page*4 && store.stats().retained_blocks==4,"Publication ignored page bound");
        check(pool->stats().live==live,"Publication allocated payload pages");
        check(!store.publish(prompt,source).inserted,"Duplicate prefix replaced its cached values");
        PrefixMatch hit=store.lookup(prompt,page*8);check(hit.entry==published.entry && hit.tokens==page*4,"Longest prefix lookup failed");
        PrefixMatch exact=store.lookup(tokens(page*4),page*8);check(exact.tokens==page*3,"Lookup did not reserve a prompt suffix");
        std::vector<int32_t> partial=prompt;partial[size_t(page+1)]+=300;
        PrefixMatch short_hit=store.lookup(partial,page*8);check(short_hit.tokens==page,"Partial token match crossed a page boundary");
        std::vector<int32_t> absent=prompt;absent[0]+=200;
        check(!store.lookup(absent,page*8).cache && !store.lookup({1},page*8).cache,"Missing prefix produced a hit");
        source.reset();equal(*hit.cache,tokens(page*4));
        append(*hit.cache,prompt);equal(*hit.cache,prompt);equal(*exact.cache,tokens(page*3));
        store.clear();check(store.stats().entries==0 && store.stats().retained_blocks==0,"Clear retained entries");
        equal(*short_hit.cache,tokens(page));equal(*hit.cache,prompt);
        hit.cache.reset();exact.cache.reset();short_hit.cache.reset();
        check(pool->stats().live==0 && pool->stats().references==0,"Eviction lost a live fork or leaked a page");
        check(!store.evict_oldest(),"Empty cache evicted an entry");
    }
    check(memory_stats(backend).live_bytes()==baseline && memory_stats(Backend::CPU).live_bytes()==host,"Prefix cache leaked tracked memory");
}
void eviction(Backend backend) {
    std::shared_ptr<elman::PagedKVPool> pool=std::make_shared<PagedKVPool>(PagedKVShape{2,1,2,4,32},backend,DType::BFloat16);
    PrefixCache store(pool,4,2);PagedKVCache a(*pool,1,32),b(*pool,1,32),c(*pool,1,32);
    std::vector<int32_t> ta=tokens(8),tb=tokens(4,100),tc=tokens(4,200);append(a,ta);append(b,tb);append(c,tc);
    store.publish(ta,a);store.publish(tb,b);ta.push_back(8);
    PrefixMatch held=store.lookup(ta,32);check(held.tokens==8,"LRU touch missed");
    check(!store.reclaim_one(),"Admission pressure discarded a fully pinned prefix");
    store.publish(tc,c);tb.push_back(104);tc.push_back(204);
    check(!store.lookup(tb,32).cache && store.lookup(tc,32).tokens==4,"Entry-count eviction ignored LRU order");
    check(store.lookup(ta,32).tokens==8,"Entry-count eviction removed the touched prefix");
    check(store.stats().entries==2 && store.stats().retained_blocks==3,"Retained page accounting differs");
    ta=tokens(16);append(a,ta);store.publish(ta,a);a.reset();b.reset();c.reset();
    check(store.stats().entries==1 && store.stats().retained_blocks==4,"Extension did not replace dominated prefixes and enforce capacity");
    equal(*held.cache,tokens(8));store.clear();equal(*held.cache,tokens(8));held.cache.reset();check(pool->stats().live==0,"Held fork leaked after final eviction");
}
void failure(Backend backend) {
    AllocationPool allocations(backend);
    std::shared_ptr<elman::PagedKVPool> pool=std::make_shared<PagedKVPool>(PagedKVShape{2,1,2,4,16},backend,DType::Float32);
    PrefixCache store(pool,4,3);PagedKVCache source(*pool,1,32);std::vector<int32_t> prompt=tokens(9);append(source,prompt);store.publish(prompt,source);
    PrefixCacheStats before=store.stats();CacheBlockStats physical=pool->stats();allocations.clear();
    {
        MemoryLimit limit(Backend::CPU,memory_stats(Backend::CPU).live_bytes());
        rejects<MemoryLimitExceeded>([&]{store.lookup(prompt,32);});
        std::vector<int32_t> different=tokens(9,100);rejects<MemoryLimitExceeded>([&]{store.publish(different,source);});
        check(!store.lookup_if_possible(prompt,32).cache && !store.publish_if_possible(different,source).inserted,
            "Optional prefix caching propagated an allocation failure");
    }
    check(store.stats().hits==before.hits && store.stats().publications==before.publications && store.stats().entries==before.entries,
        "Allocation failure mutated the prefix index");
    check(pool->stats().live==physical.live && pool->stats().references==physical.references,"Allocation failure changed page ownership");
    check(store.stats().allocation_failures==2,"Optional cache failure counters differ");
    check(store.lookup(prompt,32).tokens==8,"Prefix index did not recover");
    rejects([&]{store.lookup({},32);});rejects([&]{store.lookup(prompt,1);});rejects([&]{store.publish({1},source);});
    rejects([&]{PrefixCache invalid(pool,0,1);});rejects([&]{PrefixCache invalid(pool,17,1);});rejects([&]{PrefixCache invalid(pool,4,0);});
    std::shared_ptr<elman::PagedKVPool> other=std::make_shared<PagedKVPool>(pool->shape(),backend,DType::Float32);PagedKVCache foreign(*other,1,32),rows(*pool,2,16);
    rejects([&]{store.publish(prompt,foreign);});rejects([&]{store.publish(prompt,rows);});
    bool rejected=false;std::thread thread([&]{try{store.lookup(prompt,32);}catch(const std::invalid_argument&){rejected=true;}});thread.join();
    check(rejected,"Prefix index accepted another worker");
    source.reset();check(store.reclaim_one() && pool->stats().live==0,"Admission could not reclaim an unpinned prefix");
}
void randomized(Backend backend) {
    std::shared_ptr<elman::PagedKVPool> pool=std::make_shared<PagedKVPool>(PagedKVShape{2,1,2,3,32},backend,DType::BFloat16);PrefixCache store(pool,6,3);
    std::vector<std::vector<int32_t>> oracle;std::mt19937 rng(7019);
    for(int iteration=0;iteration<250;++iteration) {
        int family=int(rng()%5),count=3+int(rng()%9);std::vector<int32_t> prompt=tokens(count,family*100);
        if(rng()%2) {
            PagedKVCache source(*pool,1,24);append(source,prompt);std::vector<int32_t> prefix=prompt;prefix.resize(size_t(count/3*3));
            bool covered=false;for(std::vector<int>& existing:oracle)if(existing.size()>=prefix.size() && std::equal(prefix.begin(),prefix.end(),existing.begin()))covered=true;
            store.publish(prompt,source);
            if(!covered) {
                oracle.erase(std::remove_if(oracle.begin(),oracle.end(),[&](const std::vector<int32_t>& existing){return existing.size()<=prefix.size() && std::equal(existing.begin(),existing.end(),prefix.begin());}),oracle.end());
                size_t pages=prefix.size()/3;for(std::vector<int>& existing:oracle)pages+=existing.size()/3;
                while(oracle.size()==3 || pages>6) {pages-=oracle.front().size()/3;oracle.erase(oracle.begin());}
                oracle.push_back(prefix);
            }
        } else {
            size_t longest=0,index=0;
            for(size_t i=0;i<oracle.size();++i) {
                size_t common=0;while(common<oracle[i].size() && common+1<prompt.size() && oracle[i][common]==prompt[common])++common;
                common=common/3*3;if(common && common>=longest){longest=common;index=i;}
            }
            PrefixMatch hit=store.lookup(prompt,24);check(size_t(hit.tokens)==longest,"Randomized LRU prefix differs");
            if(longest) {equal(*hit.cache,{prompt.begin(),prompt.begin()+longest});std::vector<int> entry=oracle[index];oracle.erase(oracle.begin()+index);oracle.push_back(std::move(entry));}
        }
        size_t pages=0;for(std::vector<int>& entry:oracle)pages+=entry.size()/3;
        check(store.stats().entries==oracle.size() && store.stats().retained_blocks==pages,"Randomized retention bound differs");
    }
    store.clear();check(pool->stats().live==0,"Randomized prefix cache leaked pages");
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Usage: elman_prefix_cache cpu|cuda|metal");
        Backend backend=parse_backend(argv[1]);if(!backend_available(backend))return 77;
        for(elman::DType dtype:{DType::Float32,DType::Float16,DType::BFloat16})for(int page:{3,16})for(bool reuse:{false,true})lifecycle(backend,dtype,page,reuse);
        eviction(backend);failure(backend);randomized(backend);
        std::cout<<"Prefix cache "<<argv[1]<<": "<<checks<<" checks passed\n";return 0;
    }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
