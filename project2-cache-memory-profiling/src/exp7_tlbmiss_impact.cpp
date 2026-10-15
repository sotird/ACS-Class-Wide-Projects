// Experiment 7: TLB-miss impact, with 4 KiB pages and with 2 MiB huge pages.
//
// Part A (CACHE footprint held constant). Exactly L distinct cache lines are touched in every configuration; only the
// number of distinct 4 KiB pages P they are spread over changes, so only the number of TLB entries needed changes.
//   L =    512 lines ( 32 KiB, L1D-resident)  -> exposes the L1 DTLB
//   L =   8192 lines (512 KiB, L2-resident)   -> exposes the L2 DTLB
//   L =  65536 lines (  4 MiB, L3-resident)   -> larger spans
// Each configuration is run twice with the IDENTICAL layout (same virtual offsets, same visiting order): once in
// 4 KiB-page memory and once in 2 MiB-page memory (MEM_LARGE_PAGES). On 2 MiB pages the same P*4 KiB span needs
// only P/512 TLB entries (at most 128 for the largest span), so the difference between the arms is the page-size effect.
// The two arms are measured alternately, repetition by repetition, so slow drifts (clock, background load) hit both alike.
// The P page slots are randomly permuted (layout_spread(..., permute_pages=true)): on 2 MiB pages physical = virtual inside the page,
// so the plain layout would pick the L2/L3 sets itself and crowd the lines into few sets (report, pitfall 8; the unpermuted run is kept
// in data/raw/exp7_tlb_unpermuted_layout.csv). With 4 KiB pages the OS's random frames randomise those bits anyway.
//
// Part B (working-set sweep). One dependent chase over fully random lines, footprint 128 KiB ... 1 GiB, on 4 KiB and on
// 2 MiB pages: how much of the latency of a DRAM-resident chase is page-walk cost.
//
// Large pages need SeLockMemoryPrivilege ("Lock pages in memory", granted to the user by the machine's owner; this
// code only enables the already-held privilege in its own token). If it is unavailable only the 4 KiB arm runs.
//
// Usage: exp7_tlbmiss_impact.exe <output_csv_partA> <cpu> [<output_csv_partB>]
#include "chase.hpp"

using namespace bench;

// noinline on purpose: when GCC inlined this loop into main() the identical instruction (movl (%base,%off), %off) ran at 5 cycles per hop
// (0.89 ns) instead of 4 (0.71 ns, as in exp1) for L1-resident data -- a code-placement effect of the loop, not of the memory system.
// Keeping the loop in its own aligned function reproduces exp1's 0.71 ns. (Reported as a pitfall in the report.)
__attribute__((noinline)) static double timed_hops(const Chase& c, uint32_t& off, long hops) {
    auto t0 = Clock::now();
    off = chase_run(c.base, off, hops);
    return seconds_since(t0) * 1e9 / hops;
}

static bool check(const Layout& L, const Chase& c, const char* what, size_t a, size_t b) {
    if (!c.verify()) { std::cerr << "VALIDATION FAILED (" << what << ") " << a << "," << b << "\n"; return false; }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::cerr << "usage: " << argv[0] << " <output_csv> <cpu> [<output_csv_sweep>]\n"; return 1; }
    int cpu = std::atoi(argv[2]);
    init_timing_env(cpu);

    void* probe = try_alloc_huge_pages(2 * 1024 * 1024);
    const bool huge = probe != nullptr; free_pages(probe);
    std::cerr << "huge pages available: " << (huge ? "yes (2 MiB, SeLockMemoryPrivilege enabled in this process)" : "no (privilege not held or no contiguous memory): 4 KiB arm only") << "\n";

    // ---------------------------------------------------------------- Part A
    {
        CsvWriter csv(argv[1], "L_lines,P_pages,lines_per_page,span_bytes,rep,ns_per_hop,page_kind");
        struct Set { size_t L; std::vector<size_t> P; };
        std::vector<Set> sets = {
            {512,   {8, 16, 32, 64, 96, 128, 192, 256, 512}},
            {8192,  {128, 256, 512, 1024, 1536, 2048, 3072, 4096, 6144, 8192}},
            {65536, {1024, 2048, 4096, 8192, 16384, 32768, 65536}},
        };
        const int reps = 7;
        for (auto& s : sets) for (size_t P : s.P) {
            Layout L = layout_spread(s.L, P, 31337, /*permute_pages=*/true);   // balanced L2/L3 sets on 2 MiB pages (see chase.hpp)
            // lines really distinct and cache footprint constant: count distinct line addresses
            { std::vector<uint32_t> o = L.node_off; std::sort(o.begin(), o.end());
              if (std::adjacent_find(o.begin(), o.end()) != o.end()) { std::cerr << "VALIDATION FAILED: duplicate lines\n"; return 2; } }
            Chase c4; c4.build(L, false); if (!check(L, c4, "4K", s.L, P)) return 2;
            Chase c2; if (huge) { c2.build(L, true); if (!check(L, c2, "2M", s.L, P)) return 2; }
            const long warm = std::max<long>(300'000, 3 * (long)s.L), hops = 1'000'000;
            uint32_t o4 = c4.start, o2 = c2.start;
            o4 = chase_run(c4.base, o4, warm); if (huge) o2 = chase_run(c2.base, o2, warm);
            for (int r = 0; r < reps; ++r) {
                csv.row(s.L, P, s.L / P, P * 4096, r, timed_hops(c4, o4, hops), "4K");
                if (huge) csv.row(s.L, P, s.L / P, P * 4096, r, timed_hops(c2, o2, hops), "2M");
            }
            volatile uint32_t sink = o4 + o2; (void)sink;
        }
    }

    // ---------------------------------------------------------------- Part B
    if (argc >= 4) {
        CsvWriter csv(argv[3], "footprint_bytes,rep,ns_per_hop,page_kind");
        const int reps = 7;
        for (size_t kib : {128, 512, 2048, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576}) {
            size_t n = kib * 1024 / 64;
            Layout L = layout_random_lines(n, 777);
            Chase c4; c4.build(L, false); if (!check(L, c4, "ws4K", n, 0)) return 2;
            Chase c2; if (huge) { c2.build(L, true); if (!check(L, c2, "ws2M", n, 0)) return 2; }
            const long warm = std::min<long>(std::max<long>(300'000, 3 * (long)n), 6'000'000), hops = 1'000'000;
            uint32_t o4 = c4.start, o2 = c2.start;
            o4 = chase_run(c4.base, o4, warm); if (huge) o2 = chase_run(c2.base, o2, warm);
            for (int r = 0; r < reps; ++r) {
                csv.row(kib * 1024, r, timed_hops(c4, o4, hops), "4K");
                if (huge) csv.row(kib * 1024, r, timed_hops(c2, o2, hops), "2M");
            }
            volatile uint32_t sink = o4 + o2; (void)sink;
        }
    }
    std::cerr << "done: TLB-miss impact (cpu " << cpu << ", huge pages " << (huge ? "measured" : "unavailable") << ")\n";
    return 0;
}
