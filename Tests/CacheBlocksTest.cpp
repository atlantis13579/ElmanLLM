#include "Inference/LLM/CacheBlockPool.h"
#include <algorithm>
#include <atomic>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>
#include <string>
#include <thread>

using namespace elman;
namespace {
size_t checks = 0;
void check(bool valid, const char* message) {
    ++checks;
    if (!valid) throw std::runtime_error(message);
}
template<class Error, class Function> void rejects(Function function, const char* message) {
    bool rejected = false;
    try { function(); } catch (const Error&) { rejected = true; }
    check(rejected, message);
}
void boundaries() {
    rejects<std::invalid_argument>([] { CacheBlockPool pool(-1); }, "negative capacity");
    CacheBlockLease detached;
    detached.clear();
    rejects<std::logic_error>([&] { detached.resize(0); }, "detached lease cannot grow");
    CacheBlockPool empty(0);
    auto none = empty.acquire(0);
    check(none.size() == 0 && empty.stats().live == 0, "empty pool");
    rejects<CacheCapacityExceeded>([&] { empty.acquire(1); }, "zero capacity exhaustion");
    check(!empty.live({}) && !none.owns({}), "invalid default handle");

    CacheBlockPool pool(5);
    auto first = pool.acquire(3), second = pool.acquire(2);
    auto before = first.blocks(), other = second.blocks();
    check(pool.stats().live == 5 && pool.stats().peak == 5, "full pool accounting");
    try { first.resize(4); check(false, "full growth must reject"); }
    catch (const CacheCapacityExceeded& error) {
        check(error.requested() == 1 && error.available() == 0, "exhaustion details");
    }
    check(first.blocks() == before && second.blocks() == other && pool.stats().live == 5,
          "failed growth changes no ownership");
    rejects<CacheCapacityExceeded>([&] { pool.acquire(1); }, "failed new lease");
    rejects<CacheCapacityExceeded>([&] { first.resize(std::numeric_limits<size_t>::max()); }, "oversized request");
    first.resize(1);
    check(first.blocks().front() == before.front() && pool.live(before.front()), "shrink preserves prefix");
    check(!pool.live(before[1]) && !pool.live(before[2]), "shrink invalidates tail");
    auto third = pool.acquire(2);
    for (auto block : third.blocks()) {
        auto found = std::find_if(before.begin() + 1, before.end(), [&](const CacheBlock& old) { return old.index == block.index; });
        check(found != before.end() && block.generation > found->generation, "released slots receive new generations");
        check(pool.live(block) && third.owns(block) && !first.owns(block), "request ownership");
        check(!pool.live(*found), "old generation remains invalid");
    }
    CacheBlockPool foreign(5);
    auto foreign_lease = foreign.acquire(5);
    for (auto block : first.blocks()) check(!foreign.live(block), "foreign pool rejects handle");
    for (auto block : foreign_lease.blocks()) check(!pool.live(block), "pool rejects foreign handle");
    auto invalid = before.front();
    invalid.index = -1; check(!pool.live(invalid), "negative block index");
    invalid.index = 5; check(!pool.live(invalid), "block index upper bound");
    invalid = before.front(); invalid.generation = 0; check(!pool.live(invalid), "zero generation");
    first.clear(); first.clear();
    check(!pool.live(before.front()) && pool.stats().live == 4, "idempotent release");
    CacheBlockLease moved(std::move(second));
    check(second.size() == 0 && moved.blocks() == other, "move construction transfers lease");
    rejects<std::logic_error>([&] { second.resize(0); }, "moved-from lease rejects use");
    auto released = third.blocks();
    third = std::move(moved);
    check(moved.size() == 0 && third.blocks() == other && pool.stats().live == 2, "move assignment releases destination");
    for (auto block : released) check(!pool.live(block), "move assignment invalidates old destination");
    third = std::move(third);
    check(third.blocks() == other && pool.stats().live == 2, "self move preserves lease");
    third.clear();
    check(pool.stats().live == 0 && pool.stats().peak == 5, "all blocks returned");
}
void lifetime() {
    CacheBlockLease retained;
    CacheBlock old;
    {
        CacheBlockPool pool(3);
        retained = pool.acquire(2);
        old = retained.blocks().front();
    }
    retained.resize(3);
    check(retained.size() == 3 && retained.owns(old), "lease retains destroyed pool state");
    retained.clear();
    check(retained.size() == 0 && !retained.owns(old), "retained state releases safely");
    for (int i = 0; i < 100; ++i) {
        CacheBlockPool pool(1);
        auto lease = pool.acquire(1);
        check(!pool.live(old), "pool identities are not recycled");
        old = lease.blocks().front();
    }
}
void randomized() {
    CacheBlockPool pool(32);
    std::vector<CacheBlockLease> leases;
    for (int i = 0; i < 17; ++i) leases.push_back(pool.acquire(0));
    uint32_t random = 731;
    auto next = [&] { random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random; };
    for (int iteration = 0; iteration < 3000; ++iteration) {
        auto& lease = leases[next() % leases.size()];
        auto before = lease.blocks();
        size_t target = next() % 35, available = 32 - pool.stats().live;
        bool fits = target <= before.size() || target - before.size() <= available;
        try { lease.resize(target); check(fits, "growth accepted beyond capacity"); }
        catch (const CacheCapacityExceeded&) {
            check(!fits && lease.blocks() == before, "failed reservation is transactional");
        }
        if (fits) {
            check(lease.size() == target, "requested lease size");
            for (size_t i = 0; i < std::min(before.size(), target); ++i)
                check(lease.blocks()[i] == before[i], "growth and shrink retain prefix");
            for (size_t i = target; i < before.size(); ++i) check(!pool.live(before[i]), "released handle is stale");
        }
        std::set<int> occupied;
        size_t expected = 0;
        for (const auto& owner : leases) {
            expected += owner.size();
            for (auto block : owner.blocks()) {
                check(block.index >= 0 && block.index < 32, "physical block range");
                check(occupied.insert(block.index).second, "requests never share an exclusive block");
                check(pool.live(block) && owner.owns(block), "all owned handles validate");
            }
        }
        auto stats = pool.stats();
        check(stats.capacity == 32 && stats.live == expected && stats.peak >= expected && stats.peak <= 32, "independent live-block count");
    }
    leases.clear();
    check(pool.stats().live == 0, "random leases reclaim all blocks");
}
void concurrent() {
    CacheBlockPool pool(32);
    std::mutex mutex;
    std::set<int> occupied;
    std::atomic<int> failures{0}, successes{0};
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) workers.emplace_back([&, worker] {
        for (int iteration = 0; iteration < 500; ++iteration) {
            try {
                auto lease = pool.acquire(size_t(1 + (worker + iteration) % 8));
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    for (auto block : lease.blocks())
                        if (!occupied.insert(block.index).second || !pool.live(block)) ++failures;
                }
                std::this_thread::yield();
                {
                    std::lock_guard<std::mutex> lock(mutex);
                    for (auto block : lease.blocks())
                        if (occupied.erase(block.index) != 1 || !pool.live(block)) ++failures;
                }
                ++successes;
            } catch (const CacheCapacityExceeded&) {
                std::this_thread::yield();
            } catch (...) { ++failures; }
        }
    });
    for (auto& worker : workers) worker.join();
    check(failures == 0 && successes > 0 && occupied.empty() && pool.stats().live == 0, "concurrent lease isolation and reclamation");
    auto all = pool.acquire(32);
    check(all.size() == 32, "full capacity remains available after concurrency");
}
void sharing() {
    CacheBlockPool pool(5), foreign(1);
    auto first = pool.acquire(3);
    auto handles = first.blocks();
    auto child = first.share_prefix(2);
    auto empty = child.share_prefix(0);
    check(pool.stats().live == 3 && pool.stats().references == 5 && pool.stats().shared == 2,
          "shared references consume no extra physical slots");
    check(first.shared(0) && first.shared(1) && !first.shared(2), "shared ownership query");
    rejects<std::invalid_argument>([&] { first.share_prefix(4); }, "oversized shared prefix");
    rejects<std::out_of_range>([&] { first.shared(3); }, "shared query position");
    CacheBlockLease detached;
    rejects<std::logic_error>([&] { detached.share_prefix(0); }, "detached prefix sharing");
    first.clear();
    check(pool.live(handles[0]) && pool.live(handles[1]) && !pool.live(handles[2]), "only final release invalidates handles");
    check(pool.stats().live == 2 && pool.stats().references == 2 && pool.stats().shared == 0, "child retains unique references");
    auto copy = child.share_prefix(2), replacement = pool.acquire(1);
    auto fresh = replacement.blocks()[0];
    copy.swap_block(1, replacement, 0);
    check(copy.blocks()[1] == fresh && replacement.blocks()[0] == handles[1] && child.blocks()[1] == handles[1], "private replacement retains old reference");
    check(pool.stats().references == 5 && pool.stats().live == 3 && pool.stats().shared == 2, "exchange preserves reference counts");
    copy.swap_block(1, replacement, 0);
    replacement.clear();
    check(copy.blocks() == child.blocks() && !pool.live(fresh), "rollback exchanges original ownership without allocation");
    auto bad = foreign.acquire(1);
    rejects<std::invalid_argument>([&] { child.swap_block(0,bad,0); }, "foreign exchange");
    rejects<std::out_of_range>([&] { child.swap_block(2,copy,0); }, "exchange bounds");
    rejects<std::invalid_argument>([&] { child.swap_block(0,copy,1); }, "exchange would duplicate shared identities");
    child.swap_block(0,copy,0);
    child.swap_block(0,child,1); child.swap_block(0,child,1);
    child.clear(); copy.clear();
    check(pool.stats().live == 0 && pool.stats().references == 0 && pool.stats().shared == 0, "all shared leases released exactly once");
    auto reused = pool.acquire(5);
    for (auto old : handles) check(!pool.live(old), "shared handle generation is stale after reuse");

    CacheBlockLease survivor;
    {
        CacheBlockPool local(2); auto owner = local.acquire(2); survivor = owner.share_prefix(2);
    }
    check(survivor.size() == 2 && !survivor.shared(0), "shared lease survives pool and source destruction");
}
void randomized_sharing() {
    CacheBlockPool pool(16);
    std::vector<CacheBlockLease> leases;
    for (int i = 0; i < 9; ++i) leases.push_back(pool.acquire(0));
    uint32_t random = 839;
    auto next = [&] { random ^= random << 13; random ^= random >> 17; random ^= random << 5; return random; };
    for (int iteration = 0; iteration < 1500; ++iteration) {
        auto& destination = leases[next()%leases.size()];
        auto& source = leases[next()%leases.size()];
        switch (next()%3) {
        case 0: {
            auto before = destination.blocks(); size_t target = next()%12;
            bool fits = target <= before.size() || target-before.size() <= 16-pool.stats().live;
            try { destination.resize(target); check(fits, "shared growth exceeded capacity"); }
            catch (const CacheCapacityExceeded&) { check(!fits && destination.blocks()==before, "shared growth rollback"); }
            break;
        }
        case 1: destination = source.share_prefix(next()%(source.size()+1)); break;
        case 2: destination.clear(); break;
        }
        size_t references = 0, live = 0, shared = 0;
        std::vector<size_t> counts(16);
        for (const auto& lease : leases) for (auto block : lease.blocks()) {
            ++references; ++counts[size_t(block.index)]; check(pool.live(block), "shared oracle has a stale live handle");
        }
        for (auto count : counts) { live += count != 0; shared += count > 1; }
        auto actual = pool.stats();
        check(actual.live == live && actual.references == references && actual.shared == shared, "independent shared ownership oracle");
        for (const auto& lease : leases) for (size_t i = 0; i < lease.size(); ++i)
            check(lease.shared(i) == (counts[size_t(lease.blocks()[i].index)] > 1), "independent slot sharing oracle");
    }
    leases.clear();
    check(pool.stats().live == 0 && pool.stats().references == 0 && pool.stats().shared == 0, "random shared leases reclaim all slots");
}
void concurrent_sharing() {
    CacheBlockPool pool(8); auto original = pool.acquire(4);
    std::atomic<int> failures{0}; std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker) workers.emplace_back([&,worker] {
        try {
            for (int repeat = 0; repeat < 500; ++repeat) {
                auto copy = original.share_prefix(size_t(1+(repeat+worker)%4));
                auto child = copy.share_prefix(copy.size()); copy.clear();
                for (auto block : child.blocks()) if (!pool.live(block)) ++failures;
            }
        } catch (...) { ++failures; }
    });
    for (auto& thread : workers) thread.join();
    check(failures == 0 && pool.stats().live == 4 && pool.stats().references == 4 && pool.stats().shared == 0,
          "concurrent sharing retains only original references");
    original.clear(); auto all = pool.acquire(8);
    check(pool.stats().live == 8 && pool.stats().references == 8, "shared concurrent reuse retains full capacity");
}
}
int main() {
    try {
        boundaries(); lifetime(); randomized(); concurrent(); sharing(); randomized_sharing(); concurrent_sharing();
        std::cout << "Cache block ownership: " << checks << " checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Cache block ownership failed: " << error.what() << '\n';
        return 1;
    }
}
