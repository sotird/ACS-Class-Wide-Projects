// Experiment 2: access pattern and granularity sweep on a DRAM-resident
// footprint (1 GiB for latency, a private 512 MiB region per thread for
// bandwidth).
//  LATENCY (single thread, idle machine, dependent chase, 1 GiB):
//    stride 64 B / 256 B / 1024 B / 4096 B  (deterministic walk: prefetchable)
//    random line                            (unpredictable)
//  BANDWIDTH (AVX-512 64 B loads, counted as lines x 64 B actually delivered):
//    stride 64/256/1024/4096 B, random line, random BLOCK of 128..4096 B
//    (a random start, then the whole block read sequentially) with 1 thread
//    and with 8 threads (one per physical core).
//
// Usage: exp2_pattern_granularity.exe <latency_csv> <bandwidth_csv> <cpu>
#include "mt.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: " << argv[0] << " <latency_csv> <bandwidth_csv> <cpu>\n"; return 1; }
    int cpu = std::atoi(argv[3]);
    init_timing_env(cpu);
    LoadGen::tsc_ghz_ = tsc_ghz();

    // ---- validation: kernels against scalar references on a small buffer ----
    {
        std::vector<char> raw(4096 + 64);
        char* p = (char*)(((uintptr_t)raw.data() + 63) & ~(uintptr_t)63);
        std::mt19937_64 g(5); for (int i = 0; i < 4096; ++i) p[i] = (char)g();
        uint32_t ref = 0; for (int i = 0; i < 1024; ++i) { uint32_t w; std::memcpy(&w, p + 4 * i, 4); ref += w; }
        uint64_t r1 = kern::read_seq(p, 4096), r2 = kern::read_stride(p, 4096, 64);   // stride 64 == every line == sequential
        if ((uint32_t)r1 != ref || (uint32_t)r2 != ref) { std::cerr << "VALIDATION FAILED: read kernels disagree with scalar sum\n"; return 2; }
    }

    // ---------------------------- latency ----------------------------------
    {
        CsvWriter csv(argv[1], "pattern,param,rep,ns_per_hop");
        const size_t GiB = 1ull << 30;
        struct Cfg { const char* name; size_t param; Layout L; };
        std::vector<Cfg> cfgs;
        for (size_t s : {64, 256, 1024, 4096}) cfgs.push_back({"stride", s, layout_strided(GiB / s, s)});
        cfgs.push_back({"randline", 64, layout_random_lines(GiB / 64, 777)});
        for (auto& cfg : cfgs) {
            Chase c; c.build(cfg.L);
            if (!c.verify()) { std::cerr << "VALIDATION FAILED chase " << cfg.name << cfg.param << "\n"; return 2; }
            std::vector<double> all;
            chase_measure(c, 500'000, 1'500'000, 7, &all);
            for (int r = 0; r < 7; ++r) csv.row(cfg.name, cfg.param, r, all[r]);
            std::cerr << "  latency " << cfg.name << cfg.param << ": " << median(all) << " ns\n";
        }
    }

    // ---------------------------- bandwidth --------------------------------
    CsvWriter csv(argv[2], "pattern,param,threads,rep,gbps");
    LoadGen gen(8, generator_cpus(8), 1ull << 29);
    pin_thread(CONTROL_CPU);
    struct Pat { int mode; int param; };
    std::vector<Pat> pats = { {M_STRIDE, 64}, {M_STRIDE, 256}, {M_STRIDE, 1024}, {M_STRIDE, 4096}, {M_RANDLINE, 64},
                              {M_RANDBLOCK, 128}, {M_RANDBLOCK, 256}, {M_RANDBLOCK, 512}, {M_RANDBLOCK, 1024},
                              {M_RANDBLOCK, 2048}, {M_RANDBLOCK, 4096} };
    for (auto pt : pats) for (int threads : {1, 8}) for (int r = 0; r < 5; ++r) {
        gen.configure(threads, pt.mode, pt.param, 0);
        Sleep(250);
        uint64_t b0 = gen.total_bytes(); auto t0 = Clock::now();
        Sleep(500);
        uint64_t b1 = gen.total_bytes(); double dt = seconds_since(t0);
        gen.configure(0, pt.mode, pt.param, 0);
        double gbps = (b1 - b0) / dt / 1e9;
        csv.row(mode_name(pt.mode), pt.param, threads, r, gbps);
        if (r == 0) std::cerr << "  bw " << mode_name(pt.mode) << " " << pt.param << " T=" << threads << ": " << gbps << " GB/s\n";
    }
    std::cerr << "done: pattern/granularity sweep\n";
    return 0;
}
