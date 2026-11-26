#include "Inference/LLM/PagedKVCache.h"
#include "Core/LocalFunction.h"
#include "Core/AllocationPool.h"
#include "Core/Execution.h"
#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <limits>
#include <thread>

namespace {
using namespace elman;
int checks = 0;
void check(bool ok, const char* message) { ++checks; if (!ok) throw std::runtime_error(message); }
template<class E = std::invalid_argument, class F> void rejects(F action, const char* message) {
    bool caught = false;
    try { action(); } catch (const E&) { caught = true; }
    check(caught, message);
}
Tensor data(Backend backend, DType dtype, int batch, int start, int count, int layer, int tag) {
    std::vector<float> values(size_t(batch) * 2 * count * 3);
    size_t index = 0;
    for (int b = 0; b < batch; ++b) for (int h = 0; h < 2; ++h)
        for (int t = start; t < start + count; ++t) for (int d = 0; d < 3; ++d)
            values[index++] = float((tag + layer * 53 + b * 29 + h * 13 + t * 7 + d) % 251 - 125) / 8;
    return Tensor::from_data({batch, 2, count, 3}, values, backend, false, dtype);
}
void equal(const PagedKVLayer& actual, Backend backend, DType dtype, int batch, int count, int layer, int tag) {
    check(actual.key.raw_data() == data(backend, dtype, batch, 0, count, layer, tag).raw_data(), "Key payload differs");
    check(actual.value.raw_data() == data(backend, dtype, batch, 0, count, layer, tag+1).raw_data(), "Value payload differs");
}
void append(PagedKVCache& cache, int count, int tag) {
    std::unique_ptr<PagedKVAppend> transaction = cache.append(count);
    for (int layer = 0; layer < cache.shape().layers; ++layer) {
        PagedKVLayer output = transaction->write(layer,
            data(cache.backend(), cache.dtype(), cache.batch(), transaction->previous(), count, layer, tag),
            data(cache.backend(), cache.dtype(), cache.batch(), transaction->previous(), count, layer, tag+1));
        equal(output, cache.backend(), cache.dtype(), cache.batch(), transaction->total(), layer, tag);
    }
    transaction->commit();
    rejects([&] { transaction->commit(); }, "Repeated commit accepted");
}
void exercise(Backend backend, DType dtype, bool pooled, int page) {
    size_t before = memory_stats(backend).live_bytes(), host_before = memory_stats(Backend::CPU).live_bytes();
    {
        AllocationPool allocations(backend, 0, pooled);
        ExecutionStream first(backend), second(backend);
        PagedKVPool pool({3, 2, 3, page, 18}, backend, dtype);
        PagedKVCache a(pool, 2, page*5), b(pool, 1, page*5);
        check(pool.stats().live == 0 && a.length() == 0, "New cache owns blocks");
        equal(a.read(0), backend, dtype, 2, 0, 0, 7);
        rejects([&] { a.append(0); }, "Zero append accepted");
        rejects([&] { a.append(-1); }, "Negative append accepted");
        rejects([&] { a.append(page*5+1); }, "Context overflow accepted");
        {
            StreamScope scope(first);
            append(a, page, 7);
        }
        { StreamScope scope(second); append(b, page+1, 90); }
        size_t live = pool.stats().live;
        for (int written = 0; written < 3; ++written) {
            StreamScope scope(second);
            std::unique_ptr<PagedKVAppend> transaction = a.append(page+1);
            for (int layer = 0; layer < written; ++layer)
                transaction->write(layer, data(backend,dtype,2,page,page+1,layer,180), data(backend,dtype,2,page,page+1,layer,181));
            rejects([&] { transaction->commit(); }, "Incomplete append committed");
        }
        check(a.length() == page && pool.stats().live == live, "Aborted append leaked pages or length");
        for (int layer = 0; layer < 3; ++layer) {
            equal(a.read(layer), backend, dtype, 2, page, layer, 7);
            equal(b.read(layer), backend, dtype, 1, page+1, layer, 90);
        }
        // A failed mapping allocation occurs after page reservation and must
        // undo it even though the append constructor never finishes.
        {
            allocations.clear();
            size_t current = memory_stats(backend).live_bytes();
            MemoryLimit limit(backend, current);
            rejects<MemoryLimitExceeded>([&] { a.append(page+1); }, "Mapping allocation did not respect memory ceiling");
        }
        check(a.length() == page && pool.stats().live == live, "Failed constructor leaked pages");
        {
            std::unique_ptr<PagedKVAppend> transaction = a.append(1);
            Tensor good = data(backend,dtype,2,page,1,0,7);
            rejects([&] { transaction->write(-1,good,good); }, "Negative layer accepted");
            rejects([&] { transaction->write(3,good,good); }, "End layer accepted");
            rejects([&] { transaction->write(0,{},good); }, "Undefined key accepted");
            rejects([&] { transaction->write(0,good,Tensor::ones({2,2,2,3},backend,false,dtype)); }, "Bad shape accepted");
            rejects([&] { transaction->write(0,good,Tensor::ones({2,2,1,3},backend,true,dtype)); }, "Gradient payload accepted");
            PagedKVLayer written = transaction->write(0,good,data(backend,dtype,2,page,1,0,8));
            rejects([&] { transaction->write(0,good,good); }, "Duplicate layer accepted");
            { StreamScope other(second); rejects([&] { transaction->write(1,good,good); }, "Changed stream accepted"); }
        }
        check(a.length() == page && pool.stats().live == live, "Invalid write committed data");
        { StreamScope scope(first); append(a, page+1, 7); }
        PagedKVLayer snapshot = a.read(0);
        {
            std::unique_ptr<PagedKVAppend> transaction = a.append(1);
            for (int layer = 0; layer < 3; ++layer)
                transaction->write(layer, data(backend,dtype,2,page*2+1,1,layer,150), data(backend,dtype,2,page*2+1,1,layer,151));
        }
        equal(a.read(0),backend,dtype,2,page*2+1,0,7);
        a.reset();
        check(a.length() == 0 && pool.stats().live == 2, "Reset retained request blocks");
        // Released physical blocks keep buffer dependencies. Snapshot tensors
        // are independent and survive reuse by another request/stream.
        { StreamScope scope(second); append(a, page*2, 33); }
        equal(snapshot, backend,dtype,2,page*2+1,0,7);
        for (int layer = 0; layer < 3; ++layer) equal(b.read(layer),backend,dtype,1,page+1,layer,90);
        a.reset(); b.reset();
        check(pool.stats().live == 0, "Reset leaked blocks");
        for (int repeat = 0; repeat < 10; ++repeat) {
            StreamScope scope(repeat%2 ? first : second);
            append(a, 1, repeat); append(a, page, repeat);
            a.reset();
        }
        first.synchronize(); second.synchronize();
    }
    check(memory_stats(backend).live_bytes() == before && memory_stats(Backend::CPU).live_bytes() == host_before,
        "Paged cache leaked tensor or metadata memory");
}
void contracts(Backend backend) {
    for (elman::PagedKVShape shape : {PagedKVShape{0,2,3,4,5}, PagedKVShape{2,0,3,4,5}, PagedKVShape{2,2,0,4,5},
                      PagedKVShape{2,2,3,0,5}, PagedKVShape{2,2,3,4,-1}, PagedKVShape{2,2,3,4,std::numeric_limits<int>::max()}})
        rejects([&] { PagedKVPool pool(shape,backend,DType::Float32); }, "Invalid pool shape accepted");
    rejects([&] { PagedKVPool pool({2,2,3,4,5},backend,DType::Int32); }, "Integer cache accepted");
    PagedKVPool pool({2,2,3,4,3},backend,DType::Float32);
    PagedKVCache a(pool,1,16), b(pool,2,16);
    append(a,5,1);
    rejects<CacheCapacityExceeded>([&] { b.append(1); }, "Exhausted batch append succeeded");
    check(b.length() == 0 && pool.stats().live == 2, "Exhausted append partially reserved pages");
    a.reset(); append(b,4,9);
    rejects<CacheCapacityExceeded>([&] { b.append(1); }, "Growth exhaustion succeeded");
    equal(b.read(0),backend,DType::Float32,2,4,0,9);
    std::unique_ptr<PagedKVCache> retained;
    {
        PagedKVPool local({1,2,3,2,2},backend,DType::BFloat16);
        retained = std::make_unique<PagedKVCache>(local,1,4);
        append(*retained,3,20);
    }
    equal(retained->read(0),backend,DType::BFloat16,1,3,0,20);
}
void concurrent(Backend backend) {
    PagedKVPool pool({2,2,3,2,16},backend,DType::BFloat16);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 4; ++worker) threads.emplace_back([&,worker] {
        try {
            ExecutionStream stream(backend); StreamScope scope(stream);
            PagedKVCache cache(pool,1,6);
            for (int attempt = 0; attempt < 12; ++attempt) {
                for (int chunk : {1,3,2}) {
                    std::unique_ptr<PagedKVAppend> transaction = cache.append(chunk);
                    for (int layer = 0; layer < 2; ++layer) {
                        Tensor value = data(backend,DType::BFloat16,1,transaction->previous(),chunk,layer,worker*30+attempt);
                        PagedKVLayer result = transaction->write(layer,value,value);
                        if (result.key.raw_data() != data(backend,DType::BFloat16,1,0,transaction->total(),layer,worker*30+attempt).raw_data())
                            ++failures;
                    }
                    transaction->commit();
                }
                cache.reset();
            }
        } catch (...) { ++failures; }
    });
    for (std::thread& thread : threads) thread.join();
    check(failures == 0 && pool.stats().live == 0, "Concurrent requests leaked or mixed payloads");
}
void batched(Backend backend, DType dtype, bool pooled, int page) {
    size_t baseline = memory_stats(backend).tensor_bytes;
    {
        ExecutionStream stream(backend), other(backend);
        StreamScope execution(stream);
        AllocationPool allocations(backend, 0, pooled);
        PagedKVPool pool({3,2,3,page,18},backend,dtype), foreign({3,2,3,page,1},backend,dtype);
        PagedKVCache a(pool,1,page*5), b(pool,1,page*5), c(pool,1,page*5), multi(pool,2,page*5), outsider(foreign,1,page*5);
        rejects([&] { PagedKVBatch empty({}); }, "Empty batch accepted");
        rejects([&] { PagedKVBatch null({&a,nullptr}); }, "Null batch request accepted");
        rejects([&] { PagedKVBatch duplicate({&a,&a}); }, "Duplicate batch request accepted");
        rejects([&] { PagedKVBatch mixed({&a,&outsider}); }, "Foreign pool accepted");
        rejects([&] { PagedKVBatch rows({&multi}); }, "Multi-row request accepted in request batch");
        append(a,page,7); append(b,page+1,90);
        PagedKVBatch batch({&b,&c,&a});
        std::vector<int> lengths{page+1,0,page}, tags{90,150,7};
        check(batch.batch()==3 && batch.lengths()==lengths && batch.length()==page+1, "Batch lengths differ");
        size_t live = pool.stats().live;
        LocalFunction joined{[&](int layer, bool value, bool full) {
            std::vector<Tensor> rows;
            for (size_t i = 0; i < lengths.size(); ++i) {
                Tensor row = data(backend,dtype,1,full ? 0 : lengths[i],full ? lengths[i]+page : page,layer,tags[i]+int(value));
                int padding = page+1-lengths[i];
                if (full && padding) row = concat({Tensor::zeros({1,2,padding,3},backend,false,dtype),row},2);
                rows.push_back(row);
            }
            return concat(rows,0);
        }};
        for (int count : {0,1,2}) {
            std::unique_ptr<PagedKVAppend> transaction = batch.append(page);
            check(transaction->previous()==page+1 && transaction->total()==page*2+1, "Padded batch extent differs");
            for (int layer = 0; layer < count; ++layer) {
                PagedKVLayer result = transaction->write(layer,joined(layer,false,false),joined(layer,true,false));
                check(result.key.raw_data()==joined(layer,false,true).raw_data() && result.value.raw_data()==joined(layer,true,true).raw_data(),
                      "Batched gather or structural padding differs");
            }
            rejects([&] { transaction->commit(); }, "Incomplete batch committed");
        }
        check(batch.lengths()==lengths && pool.stats().live==live, "Rollback changed a participating lease");
        for (int layer = 0; layer < 3; ++layer) {
            equal(a.read(layer),backend,dtype,1,page,layer,7);
            equal(b.read(layer),backend,dtype,1,page+1,layer,90);
            equal(c.read(layer),backend,dtype,1,0,layer,150);
        }
        rejects([&] { batch.append(0); }, "Empty batched append accepted");
        rejects([&] { batch.append(page*5); }, "Per-request context overflow accepted");
        {
            std::unique_ptr<PagedKVAppend> transaction = batch.append(page);
            const std::vector<int32_t>& mask = transaction->padding_mask();
            int total = transaction->total();
            for (int row = 0; row < 3; ++row) for (int t = 0; t < total; ++t)
                check(mask[size_t(row)*total+t] == int(t >= page+1-lengths[size_t(row)]), "Structural batch mask differs");
            Tensor key = joined(0,false,false), value = joined(0,true,false);
            { StreamScope changed(other); rejects([&] { transaction->write(0,key,value); }, "Batch changed execution stream"); }
            rejects([&] { transaction->write(-1,key,value); }, "Negative batch layer accepted");
            rejects([&] { transaction->write(0,{},value); }, "Undefined batched key accepted");
            for (int layer = 0; layer < 3; ++layer) transaction->write(layer,joined(layer,false,false),joined(layer,true,false));
            rejects([&] { transaction->write(0,key,value); }, "Repeated batch layer accepted");
            transaction->commit();
            rejects([&] { transaction->commit(); }, "Repeated batch commit accepted");
        }
        check(batch.lengths()==std::vector<int>{page*2+1,page,page*2}, "Batch did not publish individual lengths");
        for (int layer = 0; layer < 3; ++layer) {
            equal(a.read(layer),backend,dtype,1,page*2,layer,7);
            equal(b.read(layer),backend,dtype,1,page*2+1,layer,90);
            equal(c.read(layer),backend,dtype,1,page,layer,150);
        }
        b.reset();
        check(a.length()==page*2 && c.length()==page, "Cancelling one row changed another request");
        a.reset(); c.reset();
        check(pool.stats().live==0, "Batched requests retained pages");
    }
    check(memory_stats(backend).tensor_bytes==baseline, "Batched payload storage leaked");
}
void batched_failures(Backend backend) {
    PagedKVPool pool({1,1,1,1,2},backend,DType::Float32);
    PagedKVCache a(pool,1,3), b(pool,1,3);
    {
        std::unique_ptr<PagedKVAppend> transaction=a.append(1);
        Tensor value=Tensor::ones({1,1,1,1},backend);
        transaction->write(0,value,value);transaction->commit();
    }
    PagedKVBatch batch({&a,&b});
    uint64_t dispatches=backend_dispatch_count(backend);
    rejects<CacheCapacityExceeded>([&] { batch.append(1); }, "Batch exhaustion did not reject");
    check(pool.stats().live==1 && a.length()==1 && b.length()==0 && backend_dispatch_count(backend)==dispatches,
          "Partial batch reservation leaked or dispatched kernels");
    a.reset();
    {
        size_t before=memory_stats(backend).live_bytes();
        MemoryLimit limit(backend,before);
        rejects<MemoryLimitExceeded>([&] { batch.append(1); }, "Batch mapping ignored memory ceiling");
    }
    check(pool.stats().live==0 && a.length()==0 && b.length()==0, "Failed batch constructor leaked pages");
    {
        std::unique_ptr<PagedKVAppend> transaction=batch.append(1);
        Tensor value=Tensor::ones({2,1,1,1},backend);
        transaction->write(0,value,value);transaction->commit();
    }
    check(pool.stats().live==2 && a.length()==1 && b.length()==1, "Batch did not recover after exhaustion");
}
void batched_nonfinite(Backend backend, DType dtype) {
    PagedKVPool pool({1,1,1,1,4},backend,dtype);
    PagedKVCache owner(pool,1,2), short_row(pool,1,2);
    {
        std::unique_ptr<PagedKVAppend> transaction=owner.append(1);
        Tensor nan=Tensor::from_data({1,1,1,1},{std::numeric_limits<float>::quiet_NaN()},backend,false,dtype);
        transaction->write(0,nan,nan); transaction->commit();
    }
    {
        PagedKVBatch batch({&owner,&short_row});
        std::unique_ptr<PagedKVAppend> transaction=batch.append(1);
        Tensor value=Tensor::ones({2,1,1,1},backend,false,dtype);
        PagedKVLayer result=transaction->write(0,value,value);
        for (const elman::Tensor* tensor : {&result.key,&result.value}) {
            std::vector<float> values=tensor->data();
            check(std::isnan(values[0]) && values[1]==1 && values[2]==0 && values[3]==1,
                  "Structural padding disclosed a nonfinite payload from another request");
        }
        transaction->commit();
    }
}
void concurrent_batches(Backend backend) {
    PagedKVPool pool({2,2,3,2,32},backend,DType::BFloat16);
    std::atomic<int> failures{0};
    std::vector<std::thread> threads;
    for (int worker=0; worker<4; ++worker) threads.emplace_back([&,worker] {
        try {
            ExecutionStream stream(backend); StreamScope scope(stream);
            PagedKVCache a(pool,1,6), b(pool,1,6);
            for (int attempt=0; attempt<12; ++attempt) {
                int tag=worker*40+attempt, lengths[2]{1,0};
                {
                    std::unique_ptr<PagedKVAppend> transaction=a.append(1);
                    for (int layer=0; layer<2; ++layer) {
                        Tensor value=data(backend,DType::BFloat16,1,0,1,layer,tag);
                        transaction->write(layer,value,value);
                    }
                    transaction->commit();
                }
                bool reverse=attempt%2!=0;
                PagedKVBatch batch(reverse ? std::vector<PagedKVCache*>{&b,&a} : std::vector<PagedKVCache*>{&a,&b});
                for (int count : {3,2}) {
                    std::unique_ptr<PagedKVAppend> transaction=batch.append(count);
                    for (int layer=0; layer<2; ++layer) {
                        std::vector<Tensor> inputs,expected;
                        for (int row=0; row<2; ++row) {
                            int request=reverse ? 1-row : row;
                            inputs.push_back(data(backend,DType::BFloat16,1,lengths[request],count,layer,tag+request*80));
                            Tensor value=data(backend,DType::BFloat16,1,0,lengths[request]+count,layer,tag+request*80);
                            if (request==1) value=concat({Tensor::zeros({1,2,1,3},backend,false,DType::BFloat16),value},2);
                            expected.push_back(value);
                        }
                        Tensor input=concat(inputs,0), target=concat(expected,0);
                        PagedKVLayer result=transaction->write(layer,input,input);
                        if (result.key.raw_data()!=target.raw_data() || result.value.raw_data()!=target.raw_data()) ++failures;
                    }
                    transaction->commit(); lengths[0]+=count; lengths[1]+=count;
                }
                a.reset(); b.reset();
            }
        } catch (...) { ++failures; }
    });
    for (std::thread& thread : threads) thread.join();
    check(failures==0 && pool.stats().live==0, "Concurrent batched requests leaked or mixed payloads");
}
void context_boundary(Backend backend) {
    size_t baseline = memory_stats(backend).tensor_bytes;
    {
        ExecutionStream stream(backend); StreamScope scope(stream);
        PagedKVShape shape{28,2,128,16,256};
        check(PagedKVPool::payload_bytes(shape,backend,DType::BFloat16) == size_t(112)*1024*1024,
            "Official 4096-token cache size differs");
        PagedKVPool pool(shape,backend,DType::BFloat16);
        PagedKVCache cache(pool,1,4096);
        for (int count : {4095,1}) {
            std::unique_ptr<PagedKVAppend> transaction = cache.append(count);
            Tensor value = Tensor::ones({1,2,count,128},backend,false,DType::BFloat16);
            for (int layer = 0; layer < 28; ++layer) {
                PagedKVLayer result = transaction->write(layer,value,value);
                check(result.key.shape() == Shape{1,2,transaction->total(),128}, "Long cache gather shape differs");
            }
            transaction->commit();
        }
        check(cache.length() == 4096 && pool.stats().live == 256, "Final cache position did not commit");
        rejects([&] { cache.append(1); }, "4096-token request accepted an extra token");
        std::vector<float> value = cache.read(27).value.data();
        check(std::all_of(value.begin(),value.end(),[](float v) { return v == 1; }), "Long cache payload differs");
        cache.reset();
        check(pool.stats().live == 0, "Long cache cancellation leaked pages");
    }
    check(memory_stats(backend).tensor_bytes == baseline, "Long cache payload leaked");
}
}
int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::invalid_argument("Usage: elman_paged_cache cpu|cuda|metal");
        Backend backend = parse_backend(argv[1]);
        if (!backend_available(backend)) return 77;
        for (elman::DType dtype : {DType::Float32,DType::Float16,DType::BFloat16})
            for (bool pooled : {false,true}) for (int page : {1,3,16}) exercise(backend,dtype,pooled,page);
        for (elman::DType dtype : {DType::Float32,DType::Float16,DType::BFloat16})
            for (bool pooled : {false,true}) for (int page : {1,3,16}) batched(backend,dtype,pooled,page);
        for (elman::DType dtype : {DType::Float32,DType::Float16,DType::BFloat16}) batched_nonfinite(backend,dtype);
        contracts(backend); concurrent(backend); concurrent_batches(backend); context_boundary(backend); batched_failures(backend);
        std::cout << "Paged cache " << argv[1] << ": " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
