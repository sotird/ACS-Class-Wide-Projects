// Pointer-chase machinery: node layouts + the dependent-load loop.
//
// A chase region holds `n` 64-byte-aligned nodes; the first 4 bytes of each
// node store the BYTE OFFSET of the next node to visit, so the loop
//     off = *(uint32_t*)(base + off)
// is one load per hop whose address depends on the previous load (a true
// serial dependency: no prefetcher or out-of-order overlap can help, so
// time/hop = load-to-use latency of whatever level holds the node).
//
// The visiting order is stored as an explicit permutation; linking node
// ord[i] -> ord[i+1] (cyclically) makes a single cycle through all nodes by
// construction. Chase::verify() re-walks the cycle and confirms it returns to
// the start after exactly n hops (no short sub-cycle).
#pragma once
#include "common.hpp"

namespace bench {

struct Layout {
    std::vector<uint32_t> node_off;   // byte offset of node j
    std::vector<uint32_t> ord;        // visiting order (permutation of node ids)
    size_t region_bytes = 0;
};

// n_lines consecutive 64 B lines, fully random visiting order. Consecutive
// hops land on unrelated pages -> also pays a TLB miss per hop once the
// footprint exceeds TLB reach.
inline Layout layout_random_lines(size_t n_lines, uint64_t seed) {
    Layout L; L.node_off.resize(n_lines); L.ord.resize(n_lines);
    for (size_t j = 0; j < n_lines; ++j) { L.node_off[j] = (uint32_t)(j * 64); L.ord[j] = (uint32_t)j; }
    std::shuffle(L.ord.begin(), L.ord.end(), std::mt19937_64(seed));
    L.region_bytes = n_lines * 64;
    return L;
}

// Same lines, but TLB-friendly: pages are visited in random order, and ALL 64
// lines of a page (in random order) are visited before moving to the next
// page. Cache behaviour is as random as above (each hop is an unpredictable
// line), but only 1 in 64 hops needs a new page translation.
inline Layout layout_page_blocked(size_t n_lines, uint64_t seed) {
    size_t pages = n_lines / 64;
    Layout L; L.node_off.resize(n_lines); L.ord.reserve(n_lines);
    for (size_t j = 0; j < n_lines; ++j) L.node_off[j] = (uint32_t)(j * 64);
    std::vector<uint32_t> pg(pages); std::iota(pg.begin(), pg.end(), 0u);
    std::mt19937_64 rng(seed);
    std::shuffle(pg.begin(), pg.end(), rng);
    std::vector<uint32_t> in_page(64);
    for (uint32_t p : pg) {
        std::iota(in_page.begin(), in_page.end(), 0u);
        std::shuffle(in_page.begin(), in_page.end(), rng);
        for (uint32_t k : in_page) L.ord.push_back(p * 64 + k);
    }
    L.region_bytes = n_lines * 64;
    return L;
}

// Fixed-stride deterministic walk: node j at j*stride_bytes, visited in order.
inline Layout layout_strided(size_t n_nodes, size_t stride_bytes) {
    Layout L; L.node_off.resize(n_nodes); L.ord.resize(n_nodes);
    for (size_t j = 0; j < n_nodes; ++j) { L.node_off[j] = (uint32_t)(j * stride_bytes); L.ord[j] = (uint32_t)j; }
    L.region_bytes = n_nodes * stride_bytes;
    return L;
}

// Controlled TLB layout: exactly L_lines distinct cache lines (so the CACHE
// footprint is constant) distributed over P pages, L_lines/P lines per page
// (<= 64), each page using a different starting line so lines of different
// pages do not all map to the same L1/L2 cache sets. Visiting order is a
// random permutation. Only the number of distinct pages (hence TLB entries
// needed) changes between configurations.
//
// permute_pages=true additionally assigns the P logical pages to RANDOMLY chosen slots of the P*4 KiB span. This matters on
// 2 MiB pages: physical = virtual within a huge page, so the bits that pick the L2/L3 set (address bits 12..15 and up) are
// exactly the virtual page-number bits, and the plain layout (page number correlated with the line offset) crowds the lines
// into 64-256 of the L2's 1024 sets. With 4 KiB pages the OS's random frame numbers randomise those bits anyway. Permuting
// the slots balances the sets in BOTH arms and leaves the number of distinct pages (the TLB footprint) unchanged.
inline Layout layout_spread(size_t L_lines, size_t P_pages, uint64_t seed, bool permute_pages = false) {
    Layout L; L.node_off.resize(L_lines); L.ord.resize(L_lines);
    std::vector<uint32_t> slot(P_pages); std::iota(slot.begin(), slot.end(), 0u);
    if (permute_pages) std::shuffle(slot.begin(), slot.end(), std::mt19937_64(seed ^ 0x9E3779B97F4A7C15ull));
    for (size_t j = 0; j < L_lines; ++j) {
        size_t page = j % P_pages, k = j / P_pages;
        // Per-page starting line chosen so every one of the 64 line offsets
        // (= every L1 set) is used about equally often for ANY P: for P < 64
        // pages the bases are spread evenly (page*64/P), for P >= 64 they wrap.
        size_t base = P_pages < 64 ? page * 64 / P_pages : page % 64;
        size_t line = (base + k) % 64;
        L.node_off[j] = (uint32_t)(slot[page] * 4096 + line * 64);
        L.ord[j] = (uint32_t)j;
    }
    std::shuffle(L.ord.begin(), L.ord.end(), std::mt19937_64(seed));
    L.region_bytes = P_pages * 4096;
    return L;
}

struct Chase {
    char* base = nullptr;
    size_t region_bytes = 0, n = 0;
    uint32_t start = 0;

    // huge=true places the region on 2 MiB pages (MEM_LARGE_PAGES); the node offsets, hence the cache-line pattern, are unchanged.
    void build(const Layout& L, bool huge = false) {
        region_bytes = L.region_bytes; n = L.ord.size();
        base = huge ? alloc_huge_pages(region_bytes) : alloc_pages(region_bytes, true);
        for (size_t i = 0; i < n; ++i) {
            uint32_t from = L.node_off[L.ord[i]], to = L.node_off[L.ord[(i + 1) % n]];
            *reinterpret_cast<uint32_t*>(base + from) = to;
        }
        start = L.node_off[L.ord[0]];
    }
    bool verify() const {
        uint32_t off = start;
        for (size_t i = 0; i < n; ++i) {
            off = *reinterpret_cast<const uint32_t*>(base + off);
            if (off == start && i + 1 < n) return false;      // closed too early: short cycle
        }
        return off == start;
    }
    ~Chase() { free_pages(base); }
};

// The timed dependent-load loop (volatile: each hop must really be loaded).
inline uint32_t chase_run(const char* base, uint32_t off, long hops) {
    for (long i = 0; i < hops; ++i) off = *reinterpret_cast<const volatile uint32_t*>(base + off);
    return off;
}

// Median-of-reps ns/hop after an untimed warm-up (a freshly-built chase region
// is biased toward whatever the build pass left in cache, so warm-up hops
// must be discarded -- see the report's honest anomalies).
struct ChaseResult { double median_ns, q1_ns, q3_ns; };
inline ChaseResult chase_measure(const Chase& c, long warmup_hops, long hops, int reps, std::vector<double>* all = nullptr) {
    uint32_t off = c.start;
    off = chase_run(c.base, off, warmup_hops);
    std::vector<double> v;
    for (int r = 0; r < reps; ++r) {
        auto t0 = Clock::now();
        off = chase_run(c.base, off, hops);
        v.push_back(seconds_since(t0) * 1e9 / hops);
    }
    volatile uint32_t sink = off; (void)sink;
    if (all) *all = v;
    return { median(v), quantile(v, 0.25), quantile(v, 0.75) };
}

} // namespace bench
