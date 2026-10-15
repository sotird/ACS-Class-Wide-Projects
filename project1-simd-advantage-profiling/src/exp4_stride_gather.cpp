// Experiment 4: stride and gather study, with the working-set CONFOUND
// removed. z[i] = x[i*S]*y[i*S] for compile-time strides S in {1,2,4,8,16},
// plus a true gather z[i] = x[idx[i]]*y[idx[i]] (random permutation).
// The source ARRAYS have a fixed length N_arr in each regime and
// n_out = N_arr / S (strided) so the footprint is identical for every
// stride; a larger stride therefore means fewer outputs from the same
// memory, i.e. each fetched cache line yields less useful data.
//   regime "L2":   N_arr = 32768  floats (x,y,idx,z <= 512 KiB: L2-resident)
//   regime "DRAM": N_arr = 16Mi   floats (x,y = 128 MiB: far beyond L3)
// BUILD_TAG: scalar | avx512 | avx512_gather (same source, different flags).
//
// Usage: exp4_stride_gather.exe <output_csv>
#ifndef BUILD_TAG
#define BUILD_TAG "unknown"
#endif
#include "common.hpp"
#include "kernels.hpp"
#include "validate.hpp"

using namespace bench;

template <size_t S>
static bool check_stride() {
    const size_t n = 257;
    std::vector<float> x(n * S + 8), y(n * S + 8), z(n, -1.f);
    fill_random(x.data(), x.size(), 5); fill_random(y.data(), y.size(), 6);
    kernels::mul_strided_c<float, S>(z.data(), x.data(), y.data(), n);
    for (size_t i = 0; i < n; ++i)
        if (z[i] != x[i * S] * y[i * S]) return false;
    return true;
}
static bool check_gather() {
    const size_t n = 1000;
    std::vector<float> x(n), y(n), z(n, -1.f);
    std::vector<int32_t> idx(n);
    fill_random(x.data(), n, 7); fill_random(y.data(), n, 8);
    std::iota(idx.begin(), idx.end(), 0);
    std::shuffle(idx.begin(), idx.end(), std::mt19937_64(9));
    kernels::mul_gather(z.data(), x.data(), y.data(), idx.data(), n);
    for (size_t i = 0; i < n; ++i)
        if (z[i] != x[idx[i]] * y[idx[i]]) return false;
    return true;
}

template <size_t S>
static void run_stride(CsvWriter& csv, const char* regime, size_t N_arr, int reps) {
    size_t n_out = N_arr / S;
    void *rx, *ry, *rz;
    float* x = aligned_alloc_offset<float>(N_arr + 64, 64, 0, &rx);
    float* y = aligned_alloc_offset<float>(N_arr + 64, 64, 0, &ry);
    float* z = aligned_alloc_offset<float>(n_out + 64, 64, 0, &rz);
    fill_random(x, N_arr + 64, 21); fill_random(y, N_arr + 64, 22); fill_random(z, n_out + 64, 23);
    auto secs = timed_reps([&]{ kernels::mul_strided_c<float, S>(z, x, y, n_out); }, reps);
    std::string pat = "stride" + std::to_string(S);
    for (int r = 0; r < reps; ++r) csv.row(regime, pat, BUILD_TAG, N_arr, n_out, r, secs[r]);
    aligned_free(rx); aligned_free(ry); aligned_free(rz);
}

static void run_gather(CsvWriter& csv, const char* regime, size_t N_arr, int reps) {
    void *rx, *ry, *rz, *ri;
    float* x = aligned_alloc_offset<float>(N_arr, 64, 0, &rx);
    float* y = aligned_alloc_offset<float>(N_arr, 64, 0, &ry);
    float* z = aligned_alloc_offset<float>(N_arr, 64, 0, &rz);
    int32_t* idx = aligned_alloc_offset<int32_t>(N_arr, 64, 0, &ri);
    fill_random(x, N_arr, 31); fill_random(y, N_arr, 32); fill_random(z, N_arr, 33);
    std::iota(idx, idx + N_arr, 0);
    std::shuffle(idx, idx + N_arr, std::mt19937_64(99));
    auto secs = timed_reps([&]{ kernels::mul_gather(z, x, y, idx, N_arr); }, reps);
    for (int r = 0; r < reps; ++r) csv.row(regime, "gather", BUILD_TAG, N_arr, N_arr, r, secs[r]);
    aligned_free(rx); aligned_free(ry); aligned_free(rz); aligned_free(ri);
}

int main(int argc, char** argv) {
    if (argc != 2) { std::cerr << "usage: " << argv[0] << " <output_csv>\n"; return 1; }
    if (!validate::all_types() || !check_stride<1>() || !check_stride<2>() || !check_stride<4>() ||
        !check_stride<8>() || !check_stride<16>() || !check_gather()) {
        std::cerr << "VALIDATION FAILED [" BUILD_TAG "]\n"; return 2;
    }
    init_timing_env();
    CsvWriter csv(argv[1], "regime,pattern,build,n_arr,n_out,rep,seconds");
    struct Regime { const char* name; size_t n; int reps; };
    for (Regime rg : { Regime{"L2", 32768, 21}, Regime{"DRAM", 16u << 20, 9} }) {
        run_stride<1>(csv, rg.name, rg.n, rg.reps);
        run_stride<2>(csv, rg.name, rg.n, rg.reps);
        run_stride<4>(csv, rg.name, rg.n, rg.reps);
        run_stride<8>(csv, rg.name, rg.n, rg.reps);
        run_stride<16>(csv, rg.name, rg.n, rg.reps);
        run_gather(csv, rg.name, rg.n, rg.reps);
    }
    std::cerr << "done: stride/gather [" BUILD_TAG "]\n";
    return 0;
}
