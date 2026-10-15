// Prints the measured sustained core clock (GHz) of the pinned logical CPU, a
// few samples apart, so run scripts can log the frequency before and after
// each experiment batch (frequency/governor discipline: Windows exposes no
// per-core governor, so the clock is MEASURED rather than assumed).
//
// Usage: freq_calibrate.exe [samples=5]    (stdout: one GHz value per line)
#include "common.hpp"
#include <cstdio>

int main(int argc, char** argv) {
    int samples = argc > 1 ? std::atoi(argv[1]) : 5;
    bench::init_timing_env();
    for (int i = 0; i < samples; ++i) {
        std::printf("%.3f\n", bench::measure_ghz(0.2));
        std::fflush(stdout);
        Sleep(100);
    }
    return 0;
}
