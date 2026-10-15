// Experiment 1: zero-queue (single outstanding request) latency vs footprint,
// L1-resident to DRAM-resident, for two chase layouts that differ ONLY in how
// many page translations they need:
//   random_lines : fully random line order (a new page almost every hop)
//   page_blocked : random pages, all 64 lines of a page (random order) before
//                  moving on (1 translation per 64 hops)
// If a latency step appears at the same footprint for both layouts it is a
// CACHE capacity effect; if it appears only for random_lines it is TLB reach.
// Single-threaded, pinned to the logical CPU given on the command line.
//
// Usage: exp1_latency_zeroqueue.exe <output_csv> <cache_info_csv> <cpu>
#include "chase.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: " << argv[0] << " <output_csv> <cache_info_csv> <cpu>\n"; return 1; }
    int cpu = std::atoi(argv[3]);
    init_timing_env(cpu);
    dump_cache_topology(argv[2]);

    // validation: the chase really is a single cycle over all nodes
    for (auto L : { layout_random_lines(4096, 1), layout_page_blocked(4096, 2), layout_spread(4096, 256, 3), layout_strided(1000, 256) }) {
        Chase c; c.build(L);
        if (!c.verify()) { std::cerr << "VALIDATION FAILED: chase is not a single cycle\n"; return 2; }
    }
    std::cerr << "chase validation passed; cpu " << cpu << ", " << measure_ghz() << " GHz\n";

    CsvWriter csv(argv[1], "cpu,layout,footprint_bytes,n_lines,rep,ns_per_hop");
    std::vector<size_t> sizes;
    for (double bytes = 4096; bytes <= 1024.0 * 1024 * 1024; bytes *= 1.15) {
        size_t n = ((size_t)(bytes / 64) / 64) * 64;               // multiple of 64 lines (whole pages)
        if (n < 64) n = 64;
        if (sizes.empty() || n != sizes.back()) sizes.push_back(n);
    }
    const int reps = 5; const long hops = 1'000'000;
    for (size_t n : sizes) {
        for (int layout = 0; layout < 2; ++layout) {
            Layout L = layout == 0 ? layout_random_lines(n, 999) : layout_page_blocked(n, 999);
            Chase c; c.build(L);
            if (!c.verify()) { std::cerr << "VALIDATION FAILED at n=" << n << "\n"; return 2; }
            std::vector<double> all;
            long warm = std::min<long>(3'000'000, std::max<long>(200'000, 2 * (long)n));
            chase_measure(c, warm, hops, reps, &all);
            for (int r = 0; r < reps; ++r)
                csv.row(cpu, layout == 0 ? "random_lines" : "page_blocked", n * 64, n, r, all[r]);
        }
    }
    std::cerr << "done: zero-queue latency sweep (cpu " << cpu << ")\n";
    return 0;
}
