// Experiment 7a: MEASURED single-core memory-bandwidth ceilings for the
// roofline, at every level of the hierarchy (L1/L2/L3/DRAM) rather than only
// DRAM. Two vectorized, associative-safe probes, swept over footprint:
//   triad : z[i] = x[i] + a*y[i]   (STREAM convention: 2 reads + 1 write, 12 B/elem)
//   sum   : s += x[i] over uint32  (read-only, 4 B/elem; integer so the compiler
//           may reassociate/vectorize it with no -ffast-math)
//
// Usage: exp7a_bandwidth_roofline.exe <output_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "validate.hpp"

using namespace bench;

static void triad(float* __restrict z, const float* __restrict x, const float* __restrict y, float a, size_t n) {
    for (size_t i = 0; i < n; ++i) z[i] = x[i] + a * y[i];
}
static uint32_t sum_u32(const uint32_t* __restrict x, size_t n) {
    uint32_t s = 0;
    for (size_t i = 0; i < n; ++i) s += x[i];
    return s;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    // validation: triad / sum against naive references at awkward sizes
    for (size_t n : {1u, 7u, 16u, 33u, 1000u, 4099u}) {
        std::vector<float> x(n), y(n), z(n), zr(n);
        fill_random(x.data(), n, 1); fill_random(y.data(), n, 2);
        triad(z.data(), x.data(), y.data(), 1.5f, n);
        for (size_t i = 0; i < n; ++i) if (std::fabs(z[i] - (x[i] + 1.5f * y[i])) > 1e-5f) { std::cerr << "VALIDATION FAILED triad\n"; return 2; }
        std::vector<uint32_t> u(n); fill_random(u.data(), n, 3);
        uint32_t ref = 0; for (size_t i = 0; i < n; ++i) ref += u[i];
        if (sum_u32(u.data(), n) != ref) { std::cerr << "VALIDATION FAILED sum\n"; return 2; }
    }
    init_timing_env();
    CsvWriter csv(argv[1], "kernel,build,n,footprint_bytes,rep,seconds,gibs");
    const int reps = 9;
    volatile uint32_t sink = 0;

    for (double fp = 6.0 * 1024; fp <= 1536.0 * 1024 * 1024; fp *= 1.5) {
        // triad: 3 arrays; footprint counts all three
        size_t n = (size_t)(fp / (3 * sizeof(float))); n = (n / 16) * 16; if (n < 16) n = 16;
        void *rx, *ry, *rz;
        float* x = aligned_alloc_offset<float>(n, 64, 0, &rx);
        float* y = aligned_alloc_offset<float>(n, 64, 0, &ry);
        float* z = aligned_alloc_offset<float>(n, 64, 0, &rz);
        fill_random(x, n, 51); fill_random(y, n, 52); fill_random(z, n, 53);
        auto s1 = timed_reps([&]{ triad(z, x, y, 1.0000001f, n); }, reps, 0.01);
        for (int r = 0; r < reps; ++r)
            csv.row("triad", BUILD_TAG, n, (unsigned long long)(3 * n * sizeof(float)), r, s1[r],
                    12.0 * n / s1[r] / (1024.0 * 1024.0 * 1024.0));
        aligned_free(rx); aligned_free(ry); aligned_free(rz);

        // read-only integer sum: footprint is a single array
        size_t m = (size_t)(fp / sizeof(uint32_t)); m = (m / 16) * 16; if (m < 16) m = 16;
        void* ru; uint32_t* u = aligned_alloc_offset<uint32_t>(m, 64, 0, &ru);
        fill_random(u, m, 54);
        auto s2 = timed_reps([&]{ sink = sum_u32(u, m); }, reps, 0.01);
        for (int r = 0; r < reps; ++r)
            csv.row("sum_u32", BUILD_TAG, m, (unsigned long long)(m * sizeof(uint32_t)), r, s2[r],
                    4.0 * m / s2[r] / (1024.0 * 1024.0 * 1024.0));
        aligned_free(ru);
    }
    std::cerr << "done: bandwidth ceilings [" BUILD_TAG "]\n";
    return 0;
}
