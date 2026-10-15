// Experiment 5: data-type comparison (f32, f64, i32) for saxpy / mul / dot,
// scalar vs vector builds (BUILD_TAG), across a working-set sweep. "i32" is
// uint32_t (wrap-around is well defined, so the compiler is free to
// reassociate the integer reduction, unlike the floating-point one).
// Differences are explained with lane count (AVX-512: 16 x 32-bit, 8 x 64-bit
// lanes per register), arithmetic intensity (bytes per element) and operation
// cost (e.g. vpmulld, the 32-bit integer multiply, is much slower than a
// floating-point multiply).
//
// Usage: exp5_dtype_compare.exe <output_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "kernels.hpp"
#include "validate.hpp"

using namespace bench;

template <typename T>
static void run(const char* dtype_name, CsvWriter& csv) {
    const int reps = 11;
    for (size_t n = 1u << 10; n <= (1u << 25); n <<= 1) {
        void *rx, *ry, *rz;
        T* x = aligned_alloc_offset<T>(n, 64, 0, &rx);
        T* y = aligned_alloc_offset<T>(n, 64, 0, &ry);
        T* z = aligned_alloc_offset<T>(n, 64, 0, &rz);
        fill_random(x, n, 31); fill_random(y, n, 32); fill_random(z, n, 33);
        volatile T a_opaque = std::is_floating_point<T>::value ? T(1.0000001) : T(3);  // runtime value: stops gcc strength-reducing the multiply
        T a = a_opaque;
        volatile T sink = T(0);
        for (const char* k : {"saxpy", "mul", "dot"}) {
            std::string ks(k);
            std::vector<double> secs;
            if (ks == "saxpy")    secs = timed_reps([&]{ kernels::saxpy(y, x, a, n); }, reps);
            else if (ks == "mul") secs = timed_reps([&]{ kernels::mul(z, x, y, n); }, reps);
            else                  secs = timed_reps([&]{ sink = kernels::dot(x, y, n); }, reps);
            auto w = kernels::work(ks, n, sizeof(T));
            for (int r = 0; r < reps; ++r)
                csv.row(k, dtype_name, BUILD_TAG, n, r, secs[r], w.flops / secs[r] / 1e9, (double)n / secs[r] / 1e9);
        }
        aligned_free(rx); aligned_free(ry); aligned_free(rz);
    }
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    if (!validate::all_types()) { std::cerr << "VALIDATION FAILED [" BUILD_TAG "]\n"; return 2; }
    init_timing_env();
    CsvWriter csv(argv[1], "kernel,dtype,build,n,rep,seconds,gops,gelem_per_s");
    run<float>("f32", csv);
    run<double>("f64", csv);
    run<uint32_t>("i32", csv);
    std::cerr << "done: dtype compare [" BUILD_TAG "]\n";
    return 0;
}
