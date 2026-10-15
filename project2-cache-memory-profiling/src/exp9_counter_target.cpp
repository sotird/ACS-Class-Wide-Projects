// Target programs for the hardware-counter experiments (exp9): each invocation runs ONE access pattern with a known
// miss behaviour; tools/pmc_record.ps1 records the counters of the process, and scripts/run_counters.ps1 differences
// two runs with different iteration counts so that allocation / initialisation cancel out.
//
//   exp9_counter_target.exe chase <lines> <pages> <hops> [<pmode>]   dependent pointer chase over <lines> 64 B nodes spread over <pages> 4 KiB
//                                                           pages; pmode 0 = 4 KiB pages (default), 1 = 2 MiB huge pages, 2 = 4 KiB pages with the
//                                                           same permuted page slots as mode 1 (control)
//   exp9_counter_target.exe rand  <lines> <calls>          independent random-order line loads (AVX-512, precomputed offset table; kernels of exp6)
//   exp9_counter_target.exe seq   <lines> <calls>          independent sequential line loads (AVX-512, address order)
//   exp9_counter_target.exe saxpy|mul <floats> <reps>      y=a*x+y  /  z=x*y over arrays of <floats> elements (auto-vectorised, -O3 -march=native)
//
// Output: "units=<n> seconds=<s> ns_per_unit=<x> ..." where a unit is one hop/line load (chase, rand, seq) or one element (saxpy, mul).
// Pinned to logical CPU 4 (SMT sibling idle), high priority, memory touched before timing.
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include "kern.hpp"

// Switches on the SeLockMemoryPrivilege already granted to this user (needed for MEM_LARGE_PAGES); does not edit any policy (same helper as common.hpp).
static bool enable_lock_memory_privilege() {
    HANDLE tok;
    if (!OpenProcessToken(GetCurrentProcess(), TOKEN_ADJUST_PRIVILEGES | TOKEN_QUERY, &tok)) return false;
    TOKEN_PRIVILEGES tp{}; tp.PrivilegeCount = 1; tp.Privileges[0].Attributes = SE_PRIVILEGE_ENABLED;
    bool ok = LookupPrivilegeValueA(nullptr, "SeLockMemoryPrivilege", &tp.Privileges[0].Luid)
           && AdjustTokenPrivileges(tok, FALSE, &tp, 0, nullptr, nullptr) && GetLastError() == ERROR_SUCCESS;
    CloseHandle(tok);
    return ok;
}

static double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

static char* alloc_touch(size_t bytes) {
    char* p = (char*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    if (!p) { std::fprintf(stderr, "VirtualAlloc(%zu) failed\n", bytes); std::exit(2); }
    std::memset(p, 1, bytes);
    return p;
}

int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s chase|rand|seq|saxpy|mul ...\n", argv[0]); return 1; }
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 4);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    const std::string mode = argv[1];
    volatile uint64_t sink = 0;
    double secs = 0; uint64_t units = 0;

    if (mode == "chase") {
        const size_t lines = std::strtoull(argv[2], 0, 10), pages = std::strtoull(argv[3], 0, 10), hops = std::strtoull(argv[4], 0, 10);
        // optional 5th argument: page mode.  0 = 4 KiB pages, pages in address order (default; the layout of all earlier runs)
        //                                    1 = 2 MiB huge pages (MEM_LARGE_PAGES), page slots randomly permuted
        //                                    2 = 4 KiB pages, page slots randomly permuted (control: identical layout to mode 1)
        // On 2 MiB pages physical = virtual inside the page, so the L2/L3 set bits are the virtual page-number bits and the address-order layout
        // crowds the lines into a fraction of the sets; permuting the slots balances them (see chase.hpp, layout_spread).
        const int pmode = argc > 5 ? std::atoi(argv[5]) : 0;
        if (pages * 64 < lines) { std::fprintf(stderr, "pages*64 must be >= lines\n"); return 1; }
        char* base;
        if (pmode == 1) {
            if (!enable_lock_memory_privilege()) { std::fprintf(stderr, "SeLockMemoryPrivilege not available\n"); return 3; }
            const size_t big = GetLargePageMinimum(), bytes = ((pages * 4096 + big - 1) / big) * big;
            base = (char*)VirtualAlloc(nullptr, bytes, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
            if (!base) { std::fprintf(stderr, "large-page VirtualAlloc(%zu) failed (error %lu)\n", bytes, GetLastError()); return 3; }
            std::memset(base, 1, bytes);
        } else base = alloc_touch(pages * 4096);
        std::vector<size_t> slot(pages); std::iota(slot.begin(), slot.end(), 0);
        if (pmode != 0) { std::mt19937_64 gs(777); for (size_t i = pages - 1; i > 0; --i) std::swap(slot[i], slot[gs() % (i + 1)]); }
        std::vector<char*> node(lines);
        // layout: page(j) = j*pages/lines spreads the lines evenly over the pages; slot(j) = j%64 gives every in-page line offset the same number
        // of lines, so the L1/L2 set mapping is balanced whatever the number of pages (a first version put all lines at offset 0 when
        // pages >= lines, which made the 'TLB' experiment a cache-set-conflict experiment -- the counters revealed it)
        for (size_t j = 0; j < lines; ++j) node[j] = base + slot[(j * pages) / lines] * 4096 + (j % 64) * 64;
        std::vector<size_t> order(lines); std::iota(order.begin(), order.end(), 0);
        std::mt19937_64 g(12345);
        for (size_t i = lines - 1; i > 0; --i) std::swap(order[i], order[g() % i]);            // Sattolo: one cycle through all nodes
        for (size_t i = 0; i < lines; ++i) *(char**)node[order[i]] = node[order[(i + 1) % lines]];
        char* p = node[order[0]];
        for (size_t i = 0; i < lines * 2; ++i) p = *(char**)p;                                 // warm-up pass (untimed)
        double t0 = now_s();
        for (size_t i = 0; i < hops; ++i) p = *(char**)p;
        secs = now_s() - t0; units = hops; sink = (uint64_t)(uintptr_t)p;
    } else if (mode == "rand" || mode == "seq") {
        // the SAME kernels as exp6 (cache-miss impact): AVX-512 64-byte loads, 'seq' in address order, 'rand' in a shuffled order given by a
        // precomputed table of line offsets (address generation inside the loop would halve the memory-level parallelism)
        const size_t lines = std::strtoull(argv[2], 0, 10), calls = std::strtoull(argv[3], 0, 10);
        char* base = alloc_touch(lines * 64);
        std::vector<uint32_t> offs(lines);
        for (size_t i = 0; i < lines; ++i) offs[i] = (uint32_t)(i * 64);
        std::shuffle(offs.begin(), offs.end(), std::mt19937_64(100));
        uint64_t s = 0;
        auto run = [&](size_t n) { for (size_t k = 0; k < n; ++k) s += (mode == "rand") ? kern::read_lines_at(base, offs.data(), lines) : kern::read_seq(base, lines * 64); };
        run(1);                                                                                // warm-up (untimed)
        double t0 = now_s(); run(calls); secs = now_s() - t0; units = (uint64_t)lines * calls; sink = s;
    } else if (mode == "saxpy" || mode == "mul") {
        const size_t n = std::strtoull(argv[2], 0, 10), reps = std::strtoull(argv[3], 0, 10);
        float* x = (float*)alloc_touch(n * 4); float* y = (float*)alloc_touch(n * 4); float* z = (float*)alloc_touch(n * 4);
        for (size_t i = 0; i < n; ++i) { x[i] = 1.0f + 1e-7f * (i & 1023); y[i] = 0.5f; z[i] = 0.0f; }
        auto run = [&](size_t r) {
            for (size_t k = 0; k < r; ++k) {
                if (mode == "saxpy") { float* __restrict yy = y; const float* __restrict xx = x; for (size_t i = 0; i < n; ++i) yy[i] = 1.0001f * xx[i] + yy[i]; }
                else { float* __restrict zz = z; const float* __restrict xx = x; const float* __restrict yy = y; for (size_t i = 0; i < n; ++i) zz[i] = xx[i] * yy[i]; }
                asm volatile("" ::: "memory");
            }
        };
        run(1);
        double t0 = now_s(); run(reps); secs = now_s() - t0; units = (uint64_t)n * reps;
        sink = (uint64_t)(y[n / 2] * 1000.0f + z[n / 3] * 1000.0f);
    } else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }

    std::printf("units=%llu seconds=%.6f ns_per_unit=%.5f sink=%llu\n", (unsigned long long)units, secs, secs * 1e9 / (double)units, (unsigned long long)sink);
    return 0;
}
