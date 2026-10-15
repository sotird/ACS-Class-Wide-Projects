// Experiment 9: target program for the hardware-counter evidence (instructions, cycles, cache-line traffic).
// Runs ONE kernel at ONE size with a given number of calls; tools\pmc_record.ps1 records the counters of the process
// and scripts\run_counters.ps1 differences two runs with different call counts so that allocation/initialisation
// cancel out. Built from the same kernels.hpp as every other experiment, once per ISA variant (scripts\build.ps1).
//
// Usage: exp9_counters_<variant>.exe <saxpy|dot|mul|stencil> <n_floats> <calls>
// Output: units=<elements processed in the timed phase> seconds=<s> ns_per_unit=<x> build=<variant>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "kernels.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 4) { std::cerr << "usage: " << argv[0] << " <saxpy|dot|mul|stencil> <n> <calls>\n"; return 1; }
    const std::string k = argv[1];
    const size_t n = std::strtoull(argv[2], nullptr, 10), calls = std::strtoull(argv[3], nullptr, 10);
    init_timing_env();
    void *rx, *ry, *rz;
    float* x = aligned_alloc_offset<float>(n, 64, 0, &rx);
    float* y = aligned_alloc_offset<float>(n, 64, 0, &ry);
    float* z = aligned_alloc_offset<float>(n, 64, 0, &rz);
    fill_random(x, n, 1); fill_random(y, n, 2); std::fill(z, z + n, 0.0f);          // touch (write) all memory before timing
    volatile float sink = 0;
    float opaque = 1.0001f; asm volatile("" : "+r"(opaque));                         // run-time multiplier (not constant-folded)
    auto call = [&]() {
        if (k == "saxpy") kernels::saxpy<float>(y, x, opaque, n);
        else if (k == "dot") sink = kernels::dot<float>(x, y, n);
        else if (k == "mul") kernels::mul<float>(z, x, y, n);
        else kernels::stencil3<float>(y, x, 0.25f, 0.5f, 0.25f, n);
        asm volatile("" ::: "memory");
    };
    call();                                                                          // warm-up (untimed)
    auto t0 = Clock::now();
    for (size_t i = 0; i < calls; ++i) call();
    const double secs = seconds_since(t0);
    const double units = (double)n * (double)calls;
    std::printf("units=%.0f seconds=%.6f ns_per_unit=%.5f build=%s sink=%g\n", units, secs, secs * 1e9 / units, BUILD_TAG, (double)sink + y[n / 2] + z[n / 3]);
    aligned_free(rx); aligned_free(ry); aligned_free(rz);
    return 0;
}
