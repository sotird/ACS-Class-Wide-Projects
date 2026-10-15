// Shared timing / pinning / allocation / statistics helpers for every
// Project 2 experiment.
#pragma once
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <immintrin.h>
#include <x86intrin.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <random>
#include <string>
#include <fstream>
#include <iostream>
#include <thread>

namespace bench {

using Clock = std::chrono::steady_clock;
inline double seconds_since(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

inline int logical_cpu_count() { SYSTEM_INFO si; GetSystemInfo(&si); return (int)si.dwNumberOfProcessors; }

// Pins the calling thread to one logical CPU (only CPUs 0..63 are addressable
// through a single affinity mask; this machine has 32).
inline void pin_thread(int cpu) { SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu); }

// Controlled-timing setup: pin to `cpu`, raise process/thread priority so
// ordinary background work is pre-empted. Affects only this process.
inline void init_timing_env(int cpu) {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    pin_thread(cpu);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    Sleep(50);
}

// Sustained core clock (GHz) from a chain of dependent 1-cycle integer adds.
inline double measure_ghz(double seconds = 0.15) {
    uint64_t x = 0; long long total = 0;
    auto t0 = Clock::now();
    while (seconds_since(t0) < seconds) {
        for (int i = 0; i < 100000; ++i) {
            asm volatile("add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                         "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                         "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t"
                         "add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t add $1,%0\n\t" : "+r"(x));
        }
        total += 100000LL * 16;
    }
    double e = seconds_since(t0);
    volatile uint64_t sink = x; (void)sink;
    return total / e / 1e9;
}

// TSC rate (GHz), for rdtsc-based delay injection.
inline double tsc_ghz() {
    auto t0 = Clock::now(); uint64_t c0 = __rdtsc();
    while (seconds_since(t0) < 0.1) {}
    uint64_t c1 = __rdtsc();
    return (c1 - c0) / seconds_since(t0) / 1e9;
}
inline void spin_ticks(uint64_t ticks) {
    uint64_t t0 = __rdtsc();
    while (__rdtsc() - t0 < ticks) _mm_pause();
}

// 4 KiB-page-granular anonymous memory. touch=true writes every page once so
// the timed region never pays first-touch page-fault / zero-fill cost.
inline char* alloc_pages(size_t bytes, bool touch = true) {
    char* p = (char*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) { std::cerr << "VirtualAlloc(" << bytes << ") failed\n"; std::exit(1); }
    if (touch) std::memset(p, 0x5A, bytes);
    return p;
}
inline void free_pages(void* p) { if (p) VirtualFree(p, 0, MEM_RELEASE); }

// MEM_LARGE_PAGES (2 MiB pages) needs SeLockMemoryPrivilege. The right itself is a Local Security Policy
// assignment ("Lock pages in memory") that the machine's owner granted to the user; this code never edits the policy.
// It only switches the already-held privilege on in this process's own token (it is listed but Disabled by default).
// Returns nullptr when large pages are unavailable (privilege not held, or no 2 MiB-contiguous physical memory), so callers can report it.
inline bool enable_lock_memory_privilege() {
    HANDLE tok;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
    TOKEN_PRIVILEGES tp{}; tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)
           && AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS;   // ERROR_NOT_ALL_ASSIGNED = not held
    CloseHandle(tok);
    return ok;
}
inline void* try_alloc_huge_pages(size_t bytes) {
    SIZE_T minLarge = GetLargePageMinimum();
    if (minLarge == 0 || !enable_lock_memory_privilege()) return nullptr;
    size_t rounded = ((bytes + minLarge - 1) / minLarge) * minLarge;
    return VirtualAlloc(nullptr, rounded, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
}
// Large-page memory, zero-filled by the OS at allocation and touched once here; exits with a message if unavailable.
inline char* alloc_huge_pages(size_t bytes) {
    char* p = (char*)try_alloc_huge_pages(bytes);
    if (!p) { std::cerr << "large-page allocation of " << bytes << " bytes failed (error " << GetLastError() << ")\n"; std::exit(1); }
    std::memset(p, 0x5A, bytes);
    return p;
}

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end()); size_t n = v.size();
    return n % 2 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
inline double quantile(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v[std::min(v.size() - 1, (size_t)(q * (v.size() - 1) + 0.5))];
}

// Doubles the inner iteration count until one batch lasts >= min_time_s, then
// returns per-call seconds for each of `reps` batches.
template <typename F>
std::vector<double> timed_reps(F&& fn, int reps, double min_time_s = 0.008) {
    std::vector<double> out; int iters = 1;
    for (int r = 0; r < reps; ++r) {
        double elapsed;
        for (;;) {
            auto t0 = Clock::now();
            for (int i = 0; i < iters; ++i) fn();
            elapsed = seconds_since(t0);
            if (elapsed >= min_time_s || iters > (1 << 24)) break;
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
    template <typename... Args> void row(Args&&... args) { write_row(args...); f_ << "\n"; }
private:
    template <typename T> void write_row(T&& v) { f_ << v; }
    template <typename T, typename... Rest> void write_row(T&& v, Rest&&... rest) { f_ << v << ","; write_row(rest...); }
    std::ofstream f_;
};

inline void dump_cache_topology(const std::string& path) {
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buf.data());
    GetLogicalProcessorInformation(info, &len);
    CsvWriter csv(path, "level,type,size_bytes,line_size,associativity");
    for (size_t i = 0; i < len / sizeof(*info); ++i) {
        if (info[i].Relationship != RelationCache) continue;
        const auto& c = info[i].Cache;
        const char* type = c.Type == CacheData ? "Data" : c.Type == CacheInstruction ? "Instruction"
                          : c.Type == CacheUnified ? "Unified" : "Trace";
        csv.row((int)c.Level, type, (unsigned long long)c.Size, (int)c.LineSize, (int)c.Associativity);
    }
}

} // namespace bench
