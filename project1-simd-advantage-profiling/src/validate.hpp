// Correctness validation shared by every Project 1 experiment binary. Each
// binary calls validate::all_types() before timing anything and exits
// non-zero if any kernel disagrees with an independent naive reference.
//
// Reference design: written separately from kernels.hpp (plain loops, NO
// restrict, long-double arithmetic for floating point, exact wrap-around
// arithmetic for 32-bit integers), checked at awkward sizes, mostly NOT
// multiples of any vector width (so scalar prologues/epilogues and masked
// tails are exercised), at several misaligned start offsets, and with
// sentinel "canary" values around every output array so an out-of-bounds
// vector/masked store in the compiled kernel would be caught.
#pragma once
#include "common.hpp"
#include "kernels.hpp"
#include <cstdio>
#include <limits>
#include <type_traits>

namespace validate {

template <typename T> constexpr bool is_fp = std::is_floating_point<T>::value;

// FP: within 8 ulps-of-scale (FMA contraction and association order may
// legitimately change the last bits). Integers: exact (mod 2^32).
template <typename T>
inline bool close_enough(T got, long double want, long double scale_abs) {
    if constexpr (is_fp<T>) {
        long double tol = 8.0L * std::numeric_limits<T>::epsilon() * scale_abs + 1e-30L;
        return std::fabs((long double)got - want) <= tol;
    } else {
        return (long double)got == want;
    }
}

template <typename T> inline long double ref_axpy(T a, T x, T y) {
    if constexpr (is_fp<T>) return (long double)a * x + y; else return (long double)(T)(a * x + y);
}
template <typename T> inline long double ref_mul(T x, T y) {
    if constexpr (is_fp<T>) return (long double)x * y; else return (long double)(T)(x * y);
}
template <typename T> inline long double ref_stencil(T a, T b, T c, T x0, T x1, T x2) {
    if constexpr (is_fp<T>) return (long double)a * x0 + (long double)b * x1 + (long double)c * x2;
    else return (long double)(T)(a * x0 + b * x1 + c * x2);
}

template <typename T>
bool all_kernels(const char* name) {
    const size_t PAD = 64;
    const size_t sizes[] = {1,2,3,5,7,8,9,15,16,17,31,32,33,63,64,65,100,255,257,1000,4099};
    const size_t offs_elems[] = {0, 1, 3};
    const T canary = is_fp<T> ? T(12345.5) : T(0xA5A5A5A5u);
    const T a = is_fp<T> ? T(1.5) : T(3), b = is_fp<T> ? T(1.25) : T(5), c = T(3);
    bool ok = true;
    int checks = 0;

    for (size_t n : sizes) for (size_t oe : offs_elems) {
        size_t total = n + 2 * PAD;
        void *rx, *ry, *rz, *ry0;
        T* bx = bench::aligned_alloc_offset<T>(total, 64, oe * sizeof(T), &rx);
        T* by = bench::aligned_alloc_offset<T>(total, 64, oe * sizeof(T), &ry);
        T* bz = bench::aligned_alloc_offset<T>(total, 64, oe * sizeof(T), &rz);
        T* by0 = bench::aligned_alloc_offset<T>(total, 64, 0, &ry0);
        bench::fill_random(bx, total, 1000 + n);
        bench::fill_random(by, total, 2000 + n);
        for (size_t i = 0; i < total; ++i) { by0[i] = by[i]; bz[i] = canary; }
        T* x = bx + PAD; T* y = by + PAD; T* z = bz + PAD; const T* y0 = by0 + PAD;

        // ---- saxpy (in place on y) ----
        kernels::saxpy(y, x, a, n);
        for (size_t i = 0; i < n; ++i) {
            long double scale = std::fabs((long double)a * x[i]) + std::fabs((long double)y0[i]);
            if (!close_enough<T>(y[i], ref_axpy(a, x[i], y0[i]), scale)) { ok = false; std::fprintf(stderr, "[%s] saxpy mismatch n=%zu off=%zu i=%zu\n", name, n, oe, i); break; }
        }
        for (size_t i = 0; i < PAD; ++i) if (by[i] != by0[i]) { ok = false; std::fprintf(stderr, "[%s] saxpy wrote before y n=%zu\n", name, n); break; }
        for (size_t i = PAD + n; i < total; ++i) if (by[i] != by0[i]) { ok = false; std::fprintf(stderr, "[%s] saxpy wrote past y n=%zu off=%zu\n", name, n, oe); break; }
        ++checks;

        // ---- mul ----
        kernels::mul(z, x, y0, n);
        for (size_t i = 0; i < n; ++i) {
            long double want = ref_mul(x[i], y0[i]);
            if (!close_enough<T>(z[i], want, std::fabs(want))) { ok = false; std::fprintf(stderr, "[%s] mul mismatch n=%zu off=%zu i=%zu\n", name, n, oe, i); break; }
        }
        for (size_t i = 0; i < PAD; ++i) if (bz[i] != canary) { ok = false; std::fprintf(stderr, "[%s] mul wrote before z n=%zu\n", name, n); break; }
        for (size_t i = PAD + n; i < total; ++i) if (bz[i] != canary) { ok = false; std::fprintf(stderr, "[%s] mul wrote past z n=%zu off=%zu (tail overrun)\n", name, n, oe); break; }
        ++checks;

        // ---- dot ----
        {
            T got = kernels::dot(x, y0, n);
            if constexpr (is_fp<T>) {
                long double want = 0, absum = 0;
                for (size_t i = 0; i < n; ++i) { long double p = (long double)x[i] * y0[i]; want += p; absum += std::fabs(p); }
                // classic bound |err| <= n*eps*sum|x_i y_i|, valid for ANY summation order
                long double scale = (long double)n * absum / 8.0L + absum;
                if (!close_enough<T>(got, want, scale)) { ok = false; std::fprintf(stderr, "[%s] dot mismatch n=%zu off=%zu\n", name, n, oe); }
            } else {
                T want = 0; for (size_t i = 0; i < n; ++i) want = (T)(want + x[i] * y0[i]);
                if (got != want) { ok = false; std::fprintf(stderr, "[%s] dot mismatch n=%zu off=%zu\n", name, n, oe); }
            }
            ++checks;
        }

        // ---- stencil ----
        if (n >= 3) {
            for (size_t i = 0; i < total; ++i) bz[i] = canary;
            kernels::stencil3(z, x, a, b, c, n);
            for (size_t i = 1; i + 1 < n; ++i) {
                long double scale = std::fabs((long double)a*x[i-1]) + std::fabs((long double)b*x[i]) + std::fabs((long double)c*x[i+1]);
                if (!close_enough<T>(z[i], ref_stencil(a, b, c, x[i-1], x[i], x[i+1]), scale)) { ok = false; std::fprintf(stderr, "[%s] stencil mismatch n=%zu off=%zu i=%zu\n", name, n, oe, i); break; }
            }
            if (z[0] != canary) { ok = false; std::fprintf(stderr, "[%s] stencil wrote z[0] n=%zu\n", name, n); }
            for (size_t i = n - 1; i < n + PAD; ++i) if (z[i] != canary) { ok = false; std::fprintf(stderr, "[%s] stencil wrote past n-2 (i=%zu) n=%zu off=%zu\n", name, i, n, oe); break; }
            ++checks;
        }
        bench::aligned_free(rx); bench::aligned_free(ry); bench::aligned_free(rz); bench::aligned_free(ry0);
    }
    std::fprintf(stderr, "validate<%s>: %d kernel/size/offset checks %s\n", name, checks, ok ? "PASSED" : "FAILED");
    return ok;
}

inline bool all_types() {
    bool ok = true;
    ok &= all_kernels<float>("f32");
    ok &= all_kernels<double>("f64");
    ok &= all_kernels<uint32_t>("i32");
    return ok;
}

} // namespace validate
