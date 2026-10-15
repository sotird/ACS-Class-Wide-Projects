// Shared timing, allocation, pinning, frequency-calibration and statistics
// helpers for all Project 1 experiments.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <malloc.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <fstream>
#include <iostream>

namespace bench {

using Clock = std::chrono::steady_clock;

inline double seconds_since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

// Controlled-timing setup used by every experiment: pin this thread to one
// logical CPU (default 4 = physical core 2; its SMT sibling, CPU 5, is left
// idle), raise process/thread priority so ordinary background work is
// pre-empted rather than interleaved, then wait briefly so the scheduler
// has actually moved us. Override the CPU with env var BENCH_CPU.
// None of this changes any system-wide setting; it only affects this process.
inline int init_timing_env() {
    int cpu = 4;
    if (const char* e = std::getenv("BENCH_CPU")) cpu = std::atoi(e);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    Sleep(50);
    return cpu;
}

// Estimates the sustained core clock (GHz) from a chain of dependent 64-bit
// integer adds: each `add` has a latency of exactly 1 cycle on every x86 core
// of interest, and they cannot overlap because each depends on the previous
// result, so cycles = adds executed and GHz = adds / elapsed / 1e9.
inline double measure_ghz(double seconds = 0.15) {
    uint64_t x = 0;
    long long total_adds = 0;
    auto t0 = Clock::now();
    while (seconds_since(t0) < seconds) {
        for (int i = 0; i < 100000; ++i) {
            asm volatile(
                "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                : "+r"(x));
        }
        total_adds += 100000LL * 16;
    }
    double elapsed = seconds_since(t0);
    volatile uint64_t sink = x; (void)sink;
    return total_adds / elapsed / 1e9;
}

// Allocates n*sizeof(T) bytes aligned to `alignment`, then shifts the usable
// pointer by byte_offset (for misalignment studies). *raw_out is for freeing.
template <typename T>
T* aligned_alloc_offset(size_t n, size_t alignment, size_t byte_offset, void** raw_out) {
    size_t bytes = n * sizeof(T) + byte_offset + alignment;
    void* raw = _aligned_malloc(bytes, alignment);
    if (!raw) { std::cerr << "allocation failed\n"; std::exit(1); }
    *raw_out = raw;
    return reinterpret_cast<T*>(reinterpret_cast<char*>(raw) + byte_offset);
}
inline void aligned_free(void* raw) { _aligned_free(raw); }

template <typename T>
void fill_random(T* p, size_t n, uint64_t seed) {
    std::mt19937_64 rng(seed);
    if constexpr (std::is_floating_point<T>::value) {
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<T>(dist(rng));
    } else {
        std::uniform_int_distribution<long long> dist(-1000, 1000);
        for (size_t i = 0; i < n; ++i) p[i] = static_cast<T>(dist(rng));
    }
}

// Runs `fn` repeatedly, doubling the inner iteration count until one batch
// takes at least min_time_s (so timer overhead is negligible even for ~ns
// kernels), then returns per-call seconds for each of `reps` batches.
template <typename F>
std::vector<double> timed_reps(F&& fn, int reps, double min_time_s = 0.004) {
    std::vector<double> out;
    out.reserve(reps);
    int iters = 1;
    for (int r = 0; r < reps; ++r) {
        double elapsed = 0.0;
        for (;;) {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) fn();
            elapsed = seconds_since(t0);
            if (elapsed >= min_time_s || iters > (1 << 22)) break;
            iters *= 2;
        }
        out.push_back(elapsed / iters);
    }
    return out;
}

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path, const std::string& header) {
        f_.open(path, std::ios::out);
        if (!f_) { std::cerr << "cannot open " << path << "\n"; std::exit(1); }
        f_ << header << "\n";
    }
    template <typename... Args>
    void row(Args&&... args) { write_row(args...); f_ << "\n"; }
private:
    template <typename T> void write_row(T&& v) { f_ << v; }
    template <typename T, typename... Rest>
    void write_row(T&& v, Rest&&... rest) { f_ << v << ","; write_row(rest...); }
    std::ofstream f_;
};

} // namespace bench
