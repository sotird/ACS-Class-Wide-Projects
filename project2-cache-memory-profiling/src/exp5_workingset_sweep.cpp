// Experiment 5: working-set bandwidth sweep (single core), 4 KiB .. 2 GiB.
//   read_avx512 : sequential AVX-512 64 B loads, integer accumulators
//   write_avx512: sequential AVX-512 64 B regular stores
//   naive_fp_sum: PITFALL kernel -- `float s += buf[i]` compiled with default
//                 FP semantics (no -ffast-math): gcc must preserve the
//                 left-to-right addition order, so the loop is one serial
//                 chain of dependent FP adds and is bound by add latency,
//                 NOT by the memory system, at every cache-resident size.
//
// Usage: exp5_workingset_sweep.exe <output_csv> <cpu>
#include "common.hpp"
#include "kern.hpp"

using namespace bench;

__attribute__((noinline)) static float naive_fp_sum(const float* p, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += p[i];
    return s;
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: " << argv[0] << " <output_csv> <cpu>\n"; return 1; }
    int cpu = std::atoi(argv[2]);
    init_timing_env(cpu);

    // validation vs. scalar references
    {
        std::vector<char> raw(8192 + 64);
        char* p = (char*)(((uintptr_t)raw.data() + 63) & ~(uintptr_t)63);
        std::mt19937_64 g(9); for (int i = 0; i < 8192; ++i) p[i] = (char)g();
        uint32_t ref = 0; for (int i = 0; i < 2048; ++i) { uint32_t w; std::memcpy(&w, p + 4 * i, 4); ref += w; }
        if ((uint32_t)kern::read_seq(p, 8192) != ref) { std::cerr << "VALIDATION FAILED read_seq\n"; return 2; }
        std::vector<float> f(1000); for (auto& v : f) v = (float)(g() % 1000) / 7.0f;
        double d = 0; for (float v : f) d += v;
        if (std::fabs(naive_fp_sum(f.data(), f.size()) - d) > 1e-2 * std::fabs(d) / 100) { std::cerr << "VALIDATION FAILED naive_fp_sum\n"; return 2; }
        kern::write_seq(p, 4096); kern::write_seq_nt(p + 4096, 4096);
        for (int i = 0; i < 8192; ++i) if (p[i] != 1) { std::cerr << "VALIDATION FAILED write kernels\n"; return 2; }
    }

    const size_t MAXB = 2ull << 30;
    char* buf = alloc_pages(MAXB, true);
    CsvWriter csv(argv[1], "kernel,footprint_bytes,rep,gbps");
    const int reps = 7;
    volatile uint64_t sink = 0; volatile float fsink = 0;
    size_t prev = 0;
    for (double fp = 4096; fp <= (double)MAXB; fp *= 1.15) {
        size_t bytes = ((size_t)fp / 512) * 512;
        if (bytes == prev) continue; prev = bytes;
        auto s1 = timed_reps([&]{ sink = kern::read_seq(buf, bytes); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("read_avx512", bytes, r, bytes / s1[r] / 1e9);
        auto s2 = timed_reps([&]{ kern::write_seq(buf, bytes); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("write_avx512", bytes, r, bytes / s2[r] / 1e9);
        auto s3 = timed_reps([&]{ fsink = naive_fp_sum((const float*)buf, bytes / 4); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("naive_fp_sum", bytes, r, bytes / s3[r] / 1e9);
    }
    free_pages(buf);
    std::cerr << "done: working-set bandwidth sweep (cpu " << cpu << ")\n";
    return 0;
}
