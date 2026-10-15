// Experiment 2: fine-grained working-set sweep for SAXPY (f32), built for
// several ISA variants (BUILD_TAG) so the SIMD speedup at each footprint is
// computed from two runs on the IDENTICAL size grid (no interpolation).
// Also dumps this machine's cache topology (Win32) for the plot annotations.
//
// Usage: exp2_locality_sweep.exe <output_csv> <cache_info_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "kernels.hpp"
#include "validate.hpp"

using namespace bench;

static void dump_cache_topology(const std::string& path) {
    DWORD len = 0;
    GetLogicalProcessorInformation(nullptr, &len);
    std::vector<char> buf(len);
    auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION*>(buf.data());
    GetLogicalProcessorInformation(info, &len);
    size_t count = len / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
    CsvWriter csv(path, "level,type,size_bytes,line_size,associativity");
    for (size_t i = 0; i < count; ++i) {
        if (info[i].Relationship != RelationCache) continue;
        const auto& c = info[i].Cache;
        const char* type = c.Type == CacheData ? "Data" : c.Type == CacheInstruction ? "Instruction"
                          : c.Type == CacheUnified ? "Unified" : "Trace";
        csv.row((int)c.Level, type, (unsigned long long)c.Size, (int)c.LineSize, (int)c.Associativity);
    }
}

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "usage: " << argv[0] << " <output_csv> <cache_info_csv>\n"; return 1; }
    if (!validate::all_types()) { std::cerr << "VALIDATION FAILED [" BUILD_TAG "]\n"; return 2; }
    int cpu = init_timing_env();
    std::cerr << "[" BUILD_TAG "] pinned to logical CPU " << cpu << "\n";
    dump_cache_topology(argv[2]);

    CsvWriter csv(argv[1], "kernel,dtype,build,n,bytes_footprint,rep,seconds,gflops,gibs");
    // ~1.15x geometric steps from 4 KiB to 1 GiB of total x+y footprint (f32)
    std::vector<size_t> sizes;
    for (double bytes = 4.0 * 1024; bytes <= 1024.0 * 1024 * 1024; bytes *= 1.15) {
        size_t n = (size_t)(bytes / (2 * sizeof(float)));
        n = (n / 16) * 16; if (n < 16) n = 16;           // keep n a multiple of 16: no tail confound here
        if (sizes.empty() || n != sizes.back()) sizes.push_back(n);
    }
    const int reps = 11;
    for (size_t n : sizes) {
        void *rx, *ry;
        float* x = aligned_alloc_offset<float>(n, 64, 0, &rx);
        float* y = aligned_alloc_offset<float>(n, 64, 0, &ry);
        fill_random(x, n, 7); fill_random(y, n, 8);
        float a = 1.0000001f;
        auto secs = timed_reps([&]{ kernels::saxpy(y, x, a, n); }, reps);
        auto w = kernels::work("saxpy", n, sizeof(float));
        double footprint = 2.0 * n * sizeof(float);
        for (int r = 0; r < reps; ++r)
            csv.row("saxpy", "f32", BUILD_TAG, n, (unsigned long long)footprint, r, secs[r],
                    w.flops / secs[r] / 1e9, w.bytes / secs[r] / (1024.0 * 1024.0 * 1024.0));
        aligned_free(rx); aligned_free(ry);
    }
    std::cerr << "done: locality sweep [" BUILD_TAG "]\n";
    return 0;
}
