// Experiment 7b: MEASURED peak FLOP/s ceilings for the roofline, one per
// vector width, using explicit FMA intrinsics on register-resident
// accumulators (NO memory traffic in the timed loop).
//
// Why intrinsics: the previous version of this experiment used a plain C++
// loop over a 2 KiB array and reported ~43 GFLOP/s as "peak". That loop
// loads and stores every element each iteration, so it was bound by L1
// load/store throughput, not by the FMA units -- an invalid compute roof.
// Here 16 independent accumulators hide the FMA latency (>= latency x
// throughput independent chains are needed to saturate the pipes), and the
// reported figure is also converted to FLOPs per CYCLE using a frequency
// measured in the same process, so it can be checked against the
// architectural expectation (lanes x 2 flops x FMA pipes).
//
// Usage: exp7b_flops_roofline.exe <output_csv>
#include "common.hpp"
#include <immintrin.h>

using namespace bench;

constexpr int NACC = 16;

// Each function executes `iters` x NACC FMAs of the given width and returns a
// checksum (prevents dead-code elimination).
static float peak_avx512(long iters) {
    __m512 acc[NACC]; for (int k = 0; k < NACC; ++k) acc[k] = _mm512_set1_ps(1.0f + k * 1e-3f);
    const __m512 b = _mm512_set1_ps(1.0000001f), c = _mm512_set1_ps(1e-7f);
    for (long i = 0; i < iters; ++i)
        for (int k = 0; k < NACC; ++k) acc[k] = _mm512_fmadd_ps(acc[k], b, c);
    __m512 s = acc[0]; for (int k = 1; k < NACC; ++k) s = _mm512_add_ps(s, acc[k]);
    return _mm512_reduce_add_ps(s);
}
static float peak_avx2(long iters) {
    __m256 acc[NACC]; for (int k = 0; k < NACC; ++k) acc[k] = _mm256_set1_ps(1.0f + k * 1e-3f);
    const __m256 b = _mm256_set1_ps(1.0000001f), c = _mm256_set1_ps(1e-7f);
    for (long i = 0; i < iters; ++i)
        for (int k = 0; k < NACC; ++k) acc[k] = _mm256_fmadd_ps(acc[k], b, c);
    __m256 s = acc[0]; for (int k = 1; k < NACC; ++k) s = _mm256_add_ps(s, acc[k]);
    float t[8]; _mm256_storeu_ps(t, s); float r = 0; for (float v : t) r += v; return r;
}
static float peak_sse(long iters) {
    __m128 acc[NACC]; for (int k = 0; k < NACC; ++k) acc[k] = _mm_set1_ps(1.0f + k * 1e-3f);
    const __m128 b = _mm_set1_ps(1.0000001f), c = _mm_set1_ps(1e-7f);
    for (long i = 0; i < iters; ++i)
        for (int k = 0; k < NACC; ++k) acc[k] = _mm_fmadd_ps(acc[k], b, c);
    __m128 s = acc[0]; for (int k = 1; k < NACC; ++k) s = _mm_add_ps(s, acc[k]);
    float t[4]; _mm_storeu_ps(t, s); return t[0] + t[1] + t[2] + t[3];
}
static float peak_scalar(long iters) {
    __m128 acc[NACC]; for (int k = 0; k < NACC; ++k) acc[k] = _mm_set_ss(1.0f + k * 1e-3f);
    const __m128 b = _mm_set_ss(1.0000001f), c = _mm_set_ss(1e-7f);
    for (long i = 0; i < iters; ++i)
        for (int k = 0; k < NACC; ++k) acc[k] = _mm_fmadd_ss(acc[k], b, c);   // scalar FMA
    float r = 0; for (int k = 0; k < NACC; ++k) r += _mm_cvtss_f32(acc[k]); return r;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    // Validation: FMA chain has a closed form, x_{n+1} = b*x_n + c, checked
    // against a scalar double-precision recurrence for 1000 steps.
    {
        double x = 1.0; for (int i = 0; i < 1000; ++i) x = 1.0000001 * x + 1e-7;
        float got = 0;
        // peak_scalar's accumulator 0 starts at 1.0 (k=0): recompute via the function
        // on a single-accumulator shadow to avoid the checksum's other lanes.
        __m128 a0 = _mm_set_ss(1.0f); const __m128 b = _mm_set_ss(1.0000001f), c = _mm_set_ss(1e-7f);
        for (int i = 0; i < 1000; ++i) a0 = _mm_fmadd_ss(a0, b, c);
        got = _mm_cvtss_f32(a0);
        if (std::fabs(got - x) > 1e-4) { std::cerr << "VALIDATION FAILED fma chain\n"; return 2; }
    }
    int cpu = init_timing_env();
    double ghz = measure_ghz(0.3);
    std::cerr << "pinned CPU " << cpu << ", measured " << ghz << " GHz\n";

    CsvWriter csv(argv[1], "isa,lanes_f32,rep,seconds,gflops,flops_per_cycle,ghz");
    struct Isa { const char* name; int lanes; float (*fn)(long); };
    Isa isas[] = { {"scalar_fma", 1, peak_scalar}, {"sse_fma", 4, peak_sse},
                   {"avx2_fma", 8, peak_avx2},     {"avx512_fma", 16, peak_avx512} };
    const long iters = 20'000'000;
    const int reps = 9;
    volatile float sink = 0;
    for (auto& isa : isas) {
        sink = isa.fn(iters / 20);                       // warm-up
        for (int r = 0; r < reps; ++r) {
            double g0 = measure_ghz(0.05);
            auto t0 = Clock::now();
            sink = isa.fn(iters);
            double s = seconds_since(t0);
            double flops = 2.0 * isa.lanes * NACC * (double)iters;
            double ghz_now = 0.5 * (g0 + measure_ghz(0.05));
            csv.row(isa.name, isa.lanes, r, s, flops / s / 1e9, flops / s / (ghz_now * 1e9), ghz_now);
        }
    }
    std::cerr << "done: peak FLOP/s ceilings (sink=" << sink << ")\n";
    return 0;
}
