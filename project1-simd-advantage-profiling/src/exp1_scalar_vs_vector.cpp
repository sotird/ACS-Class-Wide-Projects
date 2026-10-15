// Experiment 1: scalar baseline vs auto-vectorized builds, SAME source file,
// every kernel x {f32,f64} across a working-set sweep (2^10 .. 2^26 elements).
// build.ps1 compiles this file once per ISA variant; BUILD_TAG (-D) labels the
// rows: scalar | sse2 | avx2 | avx512 | avx512_fm.
//
// Usage: exp1_scalar_vs_vector.exe <output_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "kernels.hpp"
#include "validate.hpp"

using namespace bench;

template <typename T>
static void run(const char* dtype_name, CsvWriter& csv) {
    const char* kernels_list[] = {"saxpy", "dot", "mul", "stencil"};
    const int reps = 11;
    for (size_t n = 1u << 10; n <= (1u << 26); n <<= 1) {
        void *rx, *ry, *rz;
        T* x = aligned_alloc_offset<T>(n, 64, 0, &rx);
        T* y = aligned_alloc_offset<T>(n, 64, 0, &ry);
        T* z = aligned_alloc_offset<T>(n, 64, 0, &rz);
        fill_random(x, n, 42); fill_random(y, n, 43); fill_random(z, n, 44);
        volatile T a_opaque = T(1.0000001);  // runtime value (not a folded constant)
        T a = a_opaque;
        volatile T sink = T(0);

        for (const char* k : kernels_list) {
            std::vector<double> secs;
            std::string ks(k);
            if (ks == "saxpy")        secs = timed_reps([&]{ kernels::saxpy(y, x, a, n); }, reps);
            else if (ks == "dot")     secs = timed_reps([&]{ sink = kernels::dot(x, y, n); }, reps);
            else if (ks == "mul")     secs = timed_reps([&]{ kernels::mul(z, x, y, n); }, reps);
            else                      secs = timed_reps([&]{ kernels::stencil3(z, x, a, a, a, n); }, reps);
            auto w = kernels::work(ks, n, sizeof(T));
            for (int r = 0; r < reps; ++r)
                csv.row(k, dtype_name, BUILD_TAG, n, r, secs[r], w.flops / secs[r] / 1e9,
                        w.bytes / secs[r] / (1024.0 * 1024.0 * 1024.0));
        }
        aligned_free(rx); aligned_free(ry); aligned_free(rz);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    if (!validate::all_types()) { std::cerr << "VALIDATION FAILED [" BUILD_TAG "]\n"; return 2; }
    int cpu = init_timing_env();
    std::cerr << "[" BUILD_TAG "] pinned to logical CPU " << cpu << ", ~" << measure_ghz() << " GHz\n";
    CsvWriter csv(argv[1], "kernel,dtype,build,n,rep,seconds,gflops,gibs");
    run<float>("f32", csv);
    run<double>("f64", csv);
    std::cerr << "done: exp1 [" BUILD_TAG "]\n";
    return 0;
}
