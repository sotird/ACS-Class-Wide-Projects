// pmc_validate: tiny dependent-pointer-chase with a KNOWN miss behaviour, used to validate the hardware-counter
// pipeline (tools/pmc_record.ps1 + etl_pmc.exe) before it is trusted on the real experiments.
//   pmc_validate.exe <lines> <pages> <hops>
//     lines : number of 64-byte nodes in the chase (cache footprint = lines * 64 B)
//     pages : number of 4 KiB pages the nodes are spread over (>= lines/64; more pages = more TLB reach needed)
//     hops  : dependent loads timed
// Prints "ns_per_hop=<x> hops=<n> lines=<l> pages=<p> sink=<s>".  Pinned to logical CPU 4, high priority.
#include <windows.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    if (argc != 4) { std::fprintf(stderr, "usage: %s <lines> <pages> <hops>\n", argv[0]); return 1; }
    const size_t lines = std::strtoull(argv[1], 0, 10), pages = std::strtoull(argv[2], 0, 10), hops = std::strtoull(argv[3], 0, 10);
    if (pages * 64 < lines) { std::fprintf(stderr, "pages*64 must be >= lines\n"); return 1; }
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 4);
    char* base = (char*)VirtualAlloc(nullptr, pages * 4096, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::memset(base, 0, pages * 4096);                                   // touch every page once (no first-touch faults while timing)
    std::vector<char*> node(lines);
    for (size_t j = 0; j < lines; ++j) node[j] = base + (j % pages) * 4096 + ((j / pages) % 64) * 64;   // line j -> page j%pages, slot j/pages
    std::vector<size_t> order(lines); std::iota(order.begin(), order.end(), 0);
    std::mt19937_64 g(12345);
    for (size_t i = lines - 1; i > 0; --i) std::swap(order[i], order[g() % i]);           // Sattolo: one cycle through all nodes
    for (size_t i = 0; i < lines; ++i) *(char**)node[order[i]] = node[order[(i + 1) % lines]];
    char* p = node[order[0]];
    for (size_t i = 0; i < lines * 2; ++i) p = *(char**)p;                // warm-up pass
    auto t0 = std::chrono::steady_clock::now();
    for (size_t i = 0; i < hops; ++i) p = *(char**)p;
    auto t1 = std::chrono::steady_clock::now();
    std::printf("ns_per_hop=%.3f hops=%zu lines=%zu pages=%zu sink=%llu\n", std::chrono::duration<double, std::nano>(t1 - t0).count() / hops, hops, lines, pages,
                (unsigned long long)(uintptr_t)p);
    return 0;
}
