// Experiment 3: alignment and tail-handling study for SAXPY (f32), built for
// every ISA variant (BUILD_TAG: scalar|sse2|avx2|avx512).
//
//  "align_L1"/"align_L2": n is a multiple of 16 (no tail), both arrays start
//      byte_offset bytes past a 64B boundary (0..60). L1 = 2048 elems (16 KiB
//      total), L2 = 65536 elems (512 KiB total) so the effect is isolated
//      from DRAM bandwidth.
//  "tail": arrays 64B-aligned, n = 1..130 (L1-resident, call time of a few ns
//      to a few tens of ns), so any scalar prologue/epilogue or masked-tail
//      overhead is a LARGE fraction of the call and visible, instead of being
//      amortised away as it would be at n = 65536.
//
// Usage: exp3_alignment_tail.exe <output_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "kernels.hpp"
#include "validate.hpp"

using namespace bench;

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    if (!validate::all_types()) { std::cerr << "VALIDATION FAILED [" BUILD_TAG "]\n"; return 2; }
    init_timing_env();
    CsvWriter csv(argv[1], "sweep,build,n,byte_offset,rep,seconds,gflops");
    const int reps = 21;
    const float a = 1.0000001f;

    for (auto cfg : { std::make_pair("align_L1", (size_t)2048), std::make_pair("align_L2", (size_t)65536) }) {
        size_t n = cfg.second;
        for (size_t off = 0; off < 64; off += 4) {
            void *rx, *ry;
            float* x = aligned_alloc_offset<float>(n, 64, off, &rx);
            float* y = aligned_alloc_offset<float>(n, 64, off, &ry);
            fill_random(x, n, 11); fill_random(y, n, 12);
            auto secs = timed_reps([&]{ kernels::saxpy(y, x, a, n); }, reps);
            double flops = kernels::work("saxpy", n, 4).flops;
            for (int r = 0; r < reps; ++r) csv.row(cfg.first, BUILD_TAG, n, off, r, secs[r], flops / secs[r] / 1e9);
            aligned_free(rx); aligned_free(ry);
        }
    }

    for (size_t n = 1; n <= 130; ++n) {
        void *rx, *ry;
        float* x = aligned_alloc_offset<float>(n + 64, 64, 0, &rx);
        float* y = aligned_alloc_offset<float>(n + 64, 64, 0, &ry);
        fill_random(x, n + 64, 13); fill_random(y, n + 64, 14);
        auto secs = timed_reps([&]{ kernels::saxpy(y, x, a, n); }, reps, 0.002);
        double flops = kernels::work("saxpy", n, 4).flops;
        for (int r = 0; r < reps; ++r) csv.row("tail", BUILD_TAG, n, 0, r, secs[r], flops / secs[r] / 1e9);
        aligned_free(rx); aligned_free(ry);
    }
    std::cerr << "done: alignment/tail [" BUILD_TAG "]\n";
    return 0;
}
