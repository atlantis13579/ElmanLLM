#include "Inference/LLM/PagedKVCache.h"
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <thread>

using namespace elman;
namespace {
int checks = 0;
void check(bool value, const char* message) { ++checks; if (!value) throw std::runtime_error(message); }
template<class E, class F> void rejects(F action, const char* message) {
    bool caught = false; try { action(); } catch (const E&) { caught = true; }
    check(caught, message);
}
float value(int row, int head, int token, int channel, int layer, int tag) {
    return float((tag + layer*53 + row*29 + head*13 + token*7 + channel)%251-125)/8;
}
Tensor payload(Backend backend, DType dtype, int batch, int begin, const std::vector<int>& tags, int layer, bool key) {
    std::vector<float> data;
    for (int row=0; row<batch; ++row) for (int head=0; head<2; ++head)
        for (size_t token=0; token<tags.size(); ++token) for (int channel=0; channel<3; ++channel)
            data.push_back(value(row,head,begin+int(token),channel,layer,tags[token]+int(!key)));
    return Tensor::from_data({batch,2,int(tags.size()),3},data,backend,false,dtype);
}
void equal(PagedKVCache& cache, const std::vector<int>& tags) {
    check(cache.length()==int(tags.size()),"Shared cache length differs");
    for (int layer=0; layer<cache.shape().layers; ++layer) {
        PagedKVLayer actual=cache.read(layer);
        check(actual.key.raw_data()==payload(cache.backend(),cache.dtype(),cache.batch(),0,tags,layer,true).raw_data(),"Shared key prefix or suffix differs");
        check(actual.value.raw_data()==payload(cache.backend(),cache.dtype(),cache.batch(),0,tags,layer,false).raw_data(),"Shared value prefix or suffix differs");
    }
}
void append(PagedKVCache& cache, std::vector<int>& history, int count, int tag, int written=-1) {
    std::vector<int> expected=history;expected.insert(expected.end(),size_t(count),tag);
    std::unique_ptr<PagedKVAppend> transaction=cache.append(count);
    int layers=written<0 ? cache.shape().layers : written;
    for (int layer=0; layer<layers; ++layer) {
        PagedKVLayer actual=transaction->write(layer,payload(cache.backend(),cache.dtype(),cache.batch(),int(history.size()),std::vector<int>(size_t(count),tag),layer,true),
            payload(cache.backend(),cache.dtype(),cache.batch(),int(history.size()),std::vector<int>(size_t(count),tag),layer,false));
        check(actual.key.raw_data()==payload(cache.backend(),cache.dtype(),cache.batch(),0,expected,layer,true).raw_data(),"Transactional shared key differs");
        check(actual.value.raw_data()==payload(cache.backend(),cache.dtype(),cache.batch(),0,expected,layer,false).raw_data(),"Transactional shared value differs");
    }
    if (written<0) {transaction->commit();history=std::move(expected);}
    else rejects<std::invalid_argument>([&] {transaction->commit();},"Partial shared append committed");
}
void same_stats(const CacheBlockStats& a,const CacheBlockStats& b) {
    check(a.live==b.live && a.references==b.references && a.shared==b.shared,"Shared rollback changed ownership counts");
}
void branching(Backend backend,DType dtype,bool pooled,int page,int batch) {
    size_t baseline=memory_stats(backend).live_bytes(),host_baseline=memory_stats(Backend::CPU).live_bytes();
    {
        AllocationPool allocations(backend,0,pooled);
        ExecutionStream first(backend),second(backend);
        PagedKVPool pool({3,2,3,page,64},backend,dtype);
        PagedKVCache original(pool,batch,page*6);
        std::vector<int> source;{StreamScope scope(first);append(original,source,page*2+1,7);}
        std::unique_ptr<PagedKVCache> child=original.fork(page+1,page*6), sibling=original.fork(page+1,page*6);
        std::unique_ptr<PagedKVCache> nested=child->fork(1,page*6), whole=original.fork(original.length(),page*6);
        std::vector<int> child_tags(source.begin(),source.begin()+page+1), sibling_tags=child_tags,nested_tags(1,7),whole_tags=source;
        check(pool.stats().live==size_t(3*batch),"Fork allocated physical payload pages");
        check(pool.stats().shared==size_t(3*batch),"Fork did not retain all shared pages");
        rejects<std::invalid_argument>([&] {original.fork(-1,32);},"Negative prefix accepted");
        rejects<std::invalid_argument>([&] {original.fork(original.length()+1,32);},"Uncommitted prefix accepted");
        rejects<std::invalid_argument>([&] {original.fork(2,1);},"Fork context smaller than prefix accepted");
        CacheBlockStats before=pool.stats();
        {
            allocations.clear();
            MemoryLimit limit(Backend::CPU,memory_stats(Backend::CPU).live_bytes());
            rejects<MemoryLimitExceeded>([&] {original.fork(1,page*6);},"Fork metadata ignored memory limit");
        }
        same_stats(pool.stats(),before);
        for (int written=0; written<3; ++written) {
            {StreamScope scope(second);append(*child,child_tags,page+1,99,written);}
            same_stats(pool.stats(),before);equal(original,source);equal(*child,child_tags);equal(*sibling,sibling_tags);equal(*nested,nested_tags);
        }
        {
            allocations.clear();
            size_t ordinary_metadata=size_t(batch)*(size_t(page+1)+size_t(child->length()+page+1))*8+3;
            MemoryLimit limit(Backend::CPU,memory_stats(Backend::CPU).live_bytes()+ordinary_metadata);
            rejects<MemoryLimitExceeded>([&] {child->append(page+1);},"Copy-on-write metadata ignored memory limit");
        }
        same_stats(pool.stats(),before);
        {StreamScope scope(second);append(*child,child_tags,page+1,99);}
        equal(*child,child_tags);equal(original,source);equal(*sibling,sibling_tags);equal(*nested,nested_tags);
        {StreamScope scope(first);append(original,source,2,43);}
        equal(original,source);equal(*whole,whole_tags);equal(*child,child_tags);
        {StreamScope scope(second);append(*nested,nested_tags,2,80);}
        equal(*nested,nested_tags);equal(*sibling,sibling_tags);equal(original,source);
        std::unique_ptr<PagedKVCache> aligned=original.fork(page*2,page*6);std::vector<int> aligned_tags(source.begin(),source.begin()+page*2);
        size_t live=pool.stats().live;append(*aligned,aligned_tags,1,22);
        check(pool.stats().live==live+size_t(batch),"Aligned fork copied a complete read-only page");
        equal(*aligned,aligned_tags);equal(original,source);
        std::unique_ptr<PagedKVCache> empty=original.fork(0,page*6);std::vector<int> empty_tags;append(*empty,empty_tags,1,6);equal(*empty,empty_tags);
        original.reset();sibling.reset();whole.reset();aligned.reset();empty.reset();
        equal(*child,child_tags);equal(*nested,nested_tags);
        child.reset();nested.reset();
        check(pool.stats().live==0 && pool.stats().references==0 && pool.stats().shared==0,"Branch tree leaked shared pages");
        first.synchronize();second.synchronize();
    }
    check(memory_stats(backend).live_bytes()==baseline && memory_stats(Backend::CPU).live_bytes()==host_baseline,"Shared cache leaked tracked memory");
}
void batches(Backend backend,DType dtype,int page) {
    ExecutionStream stream(backend);StreamScope execution(stream);AllocationPool allocations(backend);
    PagedKVPool pool({3,2,3,page,32},backend,dtype);PagedKVCache source(pool,1,page*5),empty(pool,1,page*5);
    std::vector<int> source_tags;append(source,source_tags,page*2+1,11);
    std::unique_ptr<PagedKVCache> a=source.fork(page+1,page*5),b=source.fork(page*2-1,page*5);
    std::vector<std::vector<int>> histories{{source_tags.begin(),source_tags.begin()+page+1},{source_tags.begin(),source_tags.begin()+page*2-1},{}};
    std::vector<PagedKVCache*> requests{a.get(),b.get(),&empty};std::vector<int> tags{71,91,133};
    CacheBlockStats before=pool.stats();
    for (int written=0; written<=3; ++written) {
        const int previous=page*2-1,total=previous+2;PagedKVBatch view(requests);
        {
            std::unique_ptr<PagedKVAppend> transaction=view.append(2);
            for (int layer=0; layer<written; ++layer) {
                std::vector<float> keys,values,expected_keys,expected_values;
                for (size_t row=0; row<requests.size(); ++row) for (int head=0; head<2; ++head) {
                    int length=int(histories[row].size()),padding=previous-length;
                    for (int token=0; token<2; ++token) for (int channel=0; channel<3; ++channel) {
                        keys.push_back(value(0,head,length+token,channel,layer,tags[row]));values.push_back(value(0,head,length+token,channel,layer,tags[row]+1));
                    }
                    for (int token=0; token<total; ++token) for (int channel=0; channel<3; ++channel) {
                        int position=token-padding,tag=position<length && position>=0 ? histories[row][size_t(position)] : tags[row];
                        expected_keys.push_back(position<0 ? 0 : value(0,head,position,channel,layer,tag));
                        expected_values.push_back(position<0 ? 0 : value(0,head,position,channel,layer,tag+1));
                    }
                }
                PagedKVLayer actual=transaction->write(layer,Tensor::from_data({3,2,2,3},keys,backend,false,dtype),Tensor::from_data({3,2,2,3},values,backend,false,dtype));
                check(actual.key.raw_data()==Tensor::from_data({3,2,total,3},expected_keys,backend,false,dtype).raw_data(),"Shared batch keys or padding differ");
                check(actual.value.raw_data()==Tensor::from_data({3,2,total,3},expected_values,backend,false,dtype).raw_data(),"Shared batch values or padding differ");
            }
            if (written==3) {transaction->commit();for (size_t row=0;row<requests.size();++row)histories[row].insert(histories[row].end(),2,tags[row]);}
            else rejects<std::invalid_argument>([&]{transaction->commit();},"Incomplete shared batch committed");
        }
        if (written<3)same_stats(pool.stats(),before);
        equal(source,source_tags);for(size_t row=0;row<requests.size();++row)equal(*requests[row],histories[row]);
    }
    source.reset();a.reset();b.reset();empty.reset();check(pool.stats().live==0 && pool.stats().references==0,"Shared batch leaked leases");
}
void exhaustion(Backend backend) {
    PagedKVPool pool({3,2,3,4,1},backend,DType::BFloat16);PagedKVCache source(pool,1,4);
    std::vector<int> tags;append(source,tags,2,10);std::unique_ptr<PagedKVCache> a=source.fork(2,4),b=source.fork(2,4);CacheBlockStats before=pool.stats();
    rejects<CacheCapacityExceeded>([&]{a->append(1);},"Shared tail overwrite succeeded without a private page");
    same_stats(pool.stats(),before);equal(source,tags);equal(*a,tags);equal(*b,tags);
    a.reset();b.reset();append(source,tags,1,20);equal(source,tags);
}
void concurrent(Backend backend) {
    PagedKVPool pool({3,2,3,4,18},backend,DType::BFloat16);PagedKVCache source(pool,1,12);
    std::vector<int> history;append(source,history,5,13);
    std::atomic<int> failures{0};std::vector<std::thread> threads;
    for(int worker=0;worker<4;++worker)threads.emplace_back([&,worker] {
        try {
            ExecutionStream stream(backend);StreamScope execution(stream);
            for(int repeat=0;repeat<24;++repeat) {
                std::unique_ptr<PagedKVCache> child=source.fork(3,12);std::vector<int> tags(3,13);tags.insert(tags.end(),2,40+worker*20+repeat);
                {
                    std::unique_ptr<PagedKVAppend> transaction=child->append(2);
                    for(int layer=0;layer<3;++layer) {
                        PagedKVLayer actual=transaction->write(layer,payload(backend,DType::BFloat16,1,3,{tags[3],tags[4]},layer,true),
                            payload(backend,DType::BFloat16,1,3,{tags[3],tags[4]},layer,false));
                        if(actual.key.raw_data()!=payload(backend,DType::BFloat16,1,0,tags,layer,true).raw_data())++failures;
                        if(actual.value.raw_data()!=payload(backend,DType::BFloat16,1,0,tags,layer,false).raw_data())++failures;
                    }
                    transaction->commit();
                }
            }
        } catch(...) {++failures;}
    });
    for(std::thread& thread:threads)thread.join();
    check(failures==0 && pool.stats().live==2 && pool.stats().references==2 && pool.stats().shared==0,"Concurrent forks leaked or mixed payloads");
    equal(source,history);
}
}
int main(int argc,char** argv) {
    try {
        if(argc!=2)throw std::invalid_argument("Usage: elman_shared_cache cpu|cuda");
        Backend backend=parse_backend(argv[1]);if(!backend_available(backend))return 77;
        for(elman::DType dtype:{DType::Float32,DType::Float16,DType::BFloat16}) {
            for(bool pooled:{false,true})for(int page:{3,16})for(int batch:{1,2})branching(backend,dtype,pooled,page,batch);
            for(int page:{3,16})batches(backend,dtype,page);
        }
        exhaustion(backend);concurrent(backend);
        std::cout<<"Shared cache "<<argv[1]<<": "<<checks<<" checks passed\n";
        return 0;
    } catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
