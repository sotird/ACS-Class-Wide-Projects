// Experiment 4: intensity sweep (loaded latency) -- throughput vs latency.
// Same machinery as exp3. Two ways to dial the load from idle to saturation:
//   "delay"   : 14 generator threads, injected spin delay per 40 KiB chunk
//               swept from 800 us (almost idle) down to 0 (full speed)
//   "threads" : 0..14 generator threads at delay 0
// for read-only traffic ("100R") and for a 70/30 read/write mix. The probe
// thread's dependent-chase latency at each point, together with the measured
// bandwidth, gives the throughput-latency curve; Little's Law
// (outstanding lines = bandwidth x latency / 64 B) is applied in the analysis.
//
// Usage: exp4_intensity_sweep.exe <output_csv>
#include "mt.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    init_timing_env(PROBE_CPU);
    LoadGen::tsc_ghz_ = tsc_ghz();

    Chase probe; probe.build(layout_random_lines((256ull << 20) / 64, 4243));
    if (!probe.verify()) { std::cerr << "VALIDATION FAILED probe chase\n"; return 2; }
    uint32_t off = probe.start;
    LoadGen gen(MAX_GEN_THREADS, generator_cpus(MAX_GEN_THREADS), 1ull << 29);
    pin_thread(PROBE_CPU);
    std::cerr << "ready; probe CPU " << PROBE_CPU << ", core " << measure_ghz() << " GHz, TSC " << LoadGen::tsc_ghz_ << " GHz\n";

    CsvWriter csv(argv[1], "mix,sweep,threads,delay_us,rep,gbps,lat_mean_ns,lat_p50_ns,lat_p5_ns,lat_p95_ns");
    const double delays[] = {800, 400, 200, 100, 50, 25, 12, 6, 3, 1.5, 0.75, 0.3, 0};
    const int reps = 4;
    for (int mode : {M_READ, M_MIX70}) {
        for (double d : delays) for (int r = 0; r < reps; ++r) {
            auto p = measure_loaded(gen, probe, off, MAX_GEN_THREADS, mode, 64, d);
            csv.row(mode_name(mode), "delay", MAX_GEN_THREADS, d, r, p.gbps, p.lat.mean_ns, p.lat.p50_ns, p.lat.p5_ns, p.lat.p95_ns);
            if (r == 0) std::cerr << "  " << mode_name(mode) << " delay=" << d << "us: " << p.gbps << " GB/s, " << p.lat.p50_ns << " ns\n";
        }
        for (int t : {0, 1, 2, 3, 4, 6, 8, 10, 12, 14}) for (int r = 0; r < reps; ++r) {
            auto p = measure_loaded(gen, probe, off, t, mode, 64, 0);
            csv.row(mode_name(mode), "threads", t, 0, r, p.gbps, p.lat.mean_ns, p.lat.p50_ns, p.lat.p5_ns, p.lat.p95_ns);
        }
    }
    std::cerr << "done: intensity sweep\n";
    return 0;
}
