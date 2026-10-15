// Experiment 3b: what makes the masked AVX-512 tail expensive?  (follow-up to exp3, report Sec. 5.1)
//
// exp3 calls saxpy(y, x, a, n) repeatedly on the SAME y, so every call loads what the previous call stored. For n = 15 or 17 the
// last lanes are written by a MASKED store, and a masked store followed by a load of the same lines may stall (store forwarding);
// that is a different cost from the extra instructions of the masked tail itself. This program separates the two:
//   rotate = 0   : the same y every call, fixed pointer (exactly as exp3)
//   rotate = 1   : the same y every call, but addressed through an array of buffers (indirect)
//   rotate = 64  : calls cycle through 64 different y buffers (64 x 64 B lines, 4 KiB total, L1-resident), so a call never reads
//                  lines stored by the immediately preceding calls (the data written 63 calls earlier are long retired)
// If the n = 15/17 penalty disappears with rotation it is a store-to-load dependency; if it stays it is the tail code itself.
// Usage: exp3b_tail_rotation.exe <output_csv>      (built for avx512, like exp3's headline build)
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
    CsvWriter csv(argv[1], "build,n,rotate,rep,seconds_per_call");
    const int reps = 21;
    const float a = 1.0000001f;
    const int R = 64;
    for (size_t n : {8, 15, 16, 17, 31, 32, 33, 47, 48, 49}) {
        void *rx, *rys[R];
        float* x = aligned_alloc_offset<float>(n + 64, 64, 0, &rx);
        fill_random(x, n + 64, 13);
        float* ys[R];
        for (int k = 0; k < R; ++k) { ys[k] = aligned_alloc_offset<float>(n + 64, 64, 0, &rys[k]); fill_random(ys[k], n + 64, 20 + k); }
        // rotate = 0 : exactly exp3 (fixed y pointer held in a register)
        { float* y0 = ys[0];
          auto secs = timed_reps([&] { kernels::saxpy(y0, x, a, n); }, reps, 0.002);
          for (int r = 0; r < reps; ++r) csv.row(BUILD_TAG, n, 0, r, secs[r]); }
        // rotate = 1 : same y, but addressed through the array of buffers (indirect, like the rotating case)
        // rotate = R : cycle through R different y buffers
        for (int rotate : {1, R}) {
            volatile int sink = 0; size_t cnt = 0;
            auto secs = timed_reps([&] { kernels::saxpy(ys[cnt % rotate], x, a, n); ++cnt; }, reps, 0.002);
            sink = (int)cnt; (void)sink;
            for (int r = 0; r < reps; ++r) csv.row(BUILD_TAG, n, rotate, r, secs[r]);
        }
        aligned_free(rx); for (int k = 0; k < R; ++k) aligned_free(rys[k]);
    }
    std::cerr << "done: tail rotation [" BUILD_TAG "]\n";
    return 0;
}
