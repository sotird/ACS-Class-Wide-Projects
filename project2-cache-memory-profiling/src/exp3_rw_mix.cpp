// Experiment 3: read/write mix sweep -- BOTH throughput and latency.
// For each mix {100% read, 100% write, 100% write with non-temporal stores,
// 70/30, 50/50} N pinned generator threads stream over private 512 MiB regions
// (AVX-512, full 64 B lines, reads and writes interleaved per 10-line group)
// while the probe thread measures dependent-chase latency on its own 256 MiB
// buffer. Throughput is APPLICATION bandwidth (bytes loaded + bytes stored).
// Regular stores also cause a read-for-ownership of each line, so the DRAM
// bus carries more than the application sees (see the report); NT stores avoid it.
//
// Usage: exp3_rw_mix.exe <output_csv>
#include "mt.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    init_timing_env(PROBE_CPU);
    LoadGen::tsc_ghz_ = tsc_ghz();

    // validation: mix kernel touches exactly R read / (10-R) written lines per group
    {
        std::vector<char> raw(640 + 64, 0);
        char* p = (char*)(((uintptr_t)raw.data() + 63) & ~(uintptr_t)63);
        kern::mix_seq<7>(p, 640);
        int written = 0; for (int k = 0; k < 10; ++k) written += (p[64 * k] != 0);
        if (written != 3) { std::cerr << "VALIDATION FAILED: 70/30 mix wrote " << written << " of 10 lines (expected 3)\n"; return 2; }
        std::fill(raw.begin(), raw.end(), 0);
        kern::mix_seq<5>(p, 640);
        written = 0; for (int k = 0; k < 10; ++k) written += (p[64 * k] != 0);
        if (written != 5) { std::cerr << "VALIDATION FAILED: 50/50 mix wrote " << written << " lines\n"; return 2; }
    }

    Chase probe; probe.build(layout_random_lines((256ull << 20) / 64, 4242));
    if (!probe.verify()) { std::cerr << "VALIDATION FAILED probe chase\n"; return 2; }
    uint32_t off = probe.start;
    LoadGen gen(MAX_GEN_THREADS, generator_cpus(MAX_GEN_THREADS), 1ull << 29);
    pin_thread(PROBE_CPU);
    std::cerr << "generators ready; probe on CPU " << PROBE_CPU << ", " << measure_ghz() << " GHz\n";

    CsvWriter csv(argv[1], "mix,threads,rep,gbps,lat_mean_ns,lat_p50_ns,lat_p5_ns,lat_p95_ns");
    // idle baseline
    for (int r = 0; r < 5; ++r) {
        auto p = measure_loaded(gen, probe, off, 0, M_READ, 64, 0, 0.1, 0.5);
        csv.row("idle", 0, r, p.gbps, p.lat.mean_ns, p.lat.p50_ns, p.lat.p5_ns, p.lat.p95_ns);
    }
    for (int mode : {M_READ, M_WRITE, M_WRITE_NT, M_MIX70, M_MIX50}) {
        for (int threads : {1, 4, MAX_GEN_THREADS}) {
            for (int r = 0; r < 5; ++r) {
                auto p = measure_loaded(gen, probe, off, threads, mode, 64, 0);
                csv.row(mode_name(mode), threads, r, p.gbps, p.lat.mean_ns, p.lat.p50_ns, p.lat.p5_ns, p.lat.p95_ns);
                if (r == 0) std::cerr << "  " << mode_name(mode) << " T=" << threads << ": " << p.gbps << " GB/s, probe " << p.lat.p50_ns << " ns\n";
            }
        }
    }
    std::cerr << "done: read/write mix\n";
    return 0;
}
