// Experiment 8 (negative example #1): a loop that LOOKS SIMD-friendly (simple
// elementwise transform, restrict pointers, no branches in the visible code)
// but that GCC does NOT auto-vectorize, because the per-element work is
// hidden behind an opaque function call (black_box, compiled in a separate
// translation unit -- see black_box.cpp) that the compiler cannot inline or
// prove side-effect-free. Built with the SAME flags as the widest-vector
// build elsewhere (-O3 -march=native), and verified via -fopt-info-vec-missed
// (scripts/verify_vectorization.ps1) to be reported as NOT vectorized. For
// contrast, "inlined" performs the identical arithmetic directly in the loop
// body (and IS vectorized), so the pair quantifies the cost of the miss.
//
// Usage: exp8_negative_novec.exe <output_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"

using namespace bench;

extern "C" float black_box(float x);

static void via_call(float* __restrict y, const float* __restrict x, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = black_box(x[i]);
}
static void inlined(float* __restrict y, const float* __restrict x, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = x[i] * 1.0000001f + 1e-7f;
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }

    // Validation: both variants must compute the same thing, at a size that
    // is not a multiple of any vector width.
    for (size_t n : {1u, 15u, 16u, 17u, 257u, 4099u}) {
        std::vector<float> x(n), ya(n), yb(n);
        fill_random(x.data(), n, 61);
        via_call(ya.data(), x.data(), n);
        inlined(yb.data(), x.data(), n);
        for (size_t i = 0; i < n; ++i)
            if (std::fabs(ya[i] - yb[i]) > 1e-6f * (1.0f + std::fabs(yb[i]))) { std::cerr << "VALIDATION FAILED\n"; return 2; }
    }
    init_timing_env();

    CsvWriter csv(argv[1], "variant,build,n,rep,seconds,gflops");
    std::vector<size_t> sizes;
    for (size_t n = 1u << 10; n <= (1u << 24); n <<= 2) sizes.push_back(n);
    const int reps = 11;
    for (size_t n : sizes) {
        void *ry, *rx;
        float* y = aligned_alloc_offset<float>(n, 64, 0, &ry);
        float* x = aligned_alloc_offset<float>(n, 64, 0, &rx);
        fill_random(x, n, 62); fill_random(y, n, 63);
        double flops = 2.0 * n;
        auto s1 = timed_reps([&]{ via_call(y, x, n); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("via_call", BUILD_TAG, n, r, s1[r], flops / s1[r] / 1e9);
        auto s2 = timed_reps([&]{ inlined(y, x, n); }, reps);
        for (int r = 0; r < reps; ++r) csv.row("inlined", BUILD_TAG, n, r, s2[r], flops / s2[r] / 1e9);
        aligned_free(ry); aligned_free(rx);
    }
    std::cerr << "done: negative example (opaque call) [" BUILD_TAG "]\n";
    return 0;
}
