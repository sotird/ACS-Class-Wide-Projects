// Experiment 6: cache-miss impact, inferred from RUNTIME. The matching hardware-counter
// measurements are made by exp9_counter_target.cpp (scripts/run_counters.ps1).
// At each footprint, over the SAME set of cache lines, three ways of touching
// every line once per pass:
//   seq_indep  : sequential order, independent loads (prefetcher-friendly)
//   rand_indep : random order, independent loads (misses overlap: memory-level
//                parallelism, MLP)
//   rand_dep   : random order, each address from the previous load (dependent
//                chase: misses serialize -> full latency each)
// ns per 64 B line for each. The analysis converts the dependent-chase curve
// into per-level hit fractions (AMAT inversion) and uses rand_dep / rand_indep
// as the achieved MLP.
//
// Usage: exp6_cachemiss_impact.exe <output_csv> <cpu>
#include "chase.hpp"
#include "kern.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: " << argv[0] << " <output_csv> <cpu>\n"; return 1; }
    int cpu = std::atoi(argv[2]);
    init_timing_env(cpu);

    // validation: random-order independent loads visit each line exactly once
    {
        size_t n = 1024; Chase c; c.build(layout_random_lines(n, 3));
        std::vector<uint32_t> offs(n); for (size_t i = 0; i < n; ++i) offs[i] = (uint32_t)(i * 64);
        std::shuffle(offs.begin(), offs.end(), std::mt19937_64(4));
        std::vector<char> seen(n, 0); for (auto o : offs) seen[o / 64]++;
        for (auto s : seen) if (s != 1) { std::cerr << "VALIDATION FAILED: random order not a permutation\n"; return 2; }
        if (!c.verify()) { std::cerr << "VALIDATION FAILED chase\n"; return 2; }
    }

    CsvWriter csv(argv[1], "pattern,footprint_bytes,n_lines,rep,ns_per_line");
    const int reps = 7;
    volatile uint64_t sink = 0;
    size_t prev = 0;
    for (double fp = 8192; fp <= 1024.0 * 1024 * 1024; fp *= 1.5) {
        size_t n = ((size_t)(fp / 64) / 64) * 64;
        if (n == prev) continue; prev = n;
        Chase c; c.build(layout_random_lines(n, 99));
        if (!c.verify()) { std::cerr << "VALIDATION FAILED\n"; return 2; }
        std::vector<uint32_t> offs(n);
        for (size_t i = 0; i < n; ++i) offs[i] = (uint32_t)(i * 64);
        std::shuffle(offs.begin(), offs.end(), std::mt19937_64(100));

        auto a = timed_reps([&]{ sink = kern::read_seq(c.base, n * 64); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("seq_indep", n * 64, n, r, a[r] * 1e9 / n);
        auto b = timed_reps([&]{ sink = kern::read_lines_at(c.base, offs.data(), n); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("rand_indep", n * 64, n, r, b[r] * 1e9 / n);
        std::vector<double> d;
        long warm = std::min<long>(3'000'000, std::max<long>(200'000, 2 * (long)n));
        chase_measure(c, warm, 1'000'000, reps, &d);
        for (int r = 0; r < reps; ++r) csv.row("rand_dep", n * 64, n, r, d[r]);
    }
    std::cerr << "done: cache-miss impact (cpu " << cpu << ")\n";
    return 0;
}
