// Prints the measured sustained clock (GHz) of the pinned logical CPU, a few
// samples apart (frequency discipline: Windows exposes no per-core governor,
// so the clock is MEASURED rather than assumed).
// Usage: freq_calibrate.exe [samples=5] [cpu=4]
#include "common.hpp"
#include <cstdio>
int main(int argc, char** argv) {
    int samples = argc > 1 ? std::atoi(argv[1]) : 5;
    int cpu = argc > 2 ? std::atoi(argv[2]) : 4;
    bench::init_timing_env(cpu);
    for (int i = 0; i < samples; ++i) { std::printf("%.3f\n", bench::measure_ghz(0.2)); std::fflush(stdout); Sleep(100); }
    return 0;
}
