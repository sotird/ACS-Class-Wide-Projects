// Kernel definitions shared by every Project 1 experiment. Every kernel is a
// plain restrict-pointer loop, so the ONLY thing that changes between the
// "scalar" and "vector" measurements is the compiler flags used to build this
// same source (see scripts/build.ps1) -- never a hand-written second path.
#pragma once
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>

namespace kernels {

// y[i] = a*x[i] + y[i]
template <typename T>
inline void saxpy(T* __restrict y, const T* __restrict x, T a, size_t n) {
    for (size_t i = 0; i < n; ++i) y[i] = a * x[i] + y[i];
}

// returns sum(x[i]*y[i]) -- NOTE: strict left-to-right FP summation order is
// part of C++ semantics; see the report's negative case (needs -ffast-math /
// -fassociative-math for gcc to reassociate it into parallel partial sums).
template <typename T>
inline T dot(const T* __restrict x, const T* __restrict y, size_t n) {
    T acc = T(0);
    for (size_t i = 0; i < n; ++i) acc += x[i] * y[i];
    return acc;
}

// z[i] = x[i]*y[i]
template <typename T>
inline void mul(T* __restrict z, const T* __restrict x, const T* __restrict y, size_t n) {
    for (size_t i = 0; i < n; ++i) z[i] = x[i] * y[i];
}

// y[i] = a*x[i-1] + b*x[i] + c*x[i+1], for i in [1, n-1)
template <typename T>
inline void stencil3(T* __restrict y, const T* __restrict x, T a, T b, T c, size_t n) {
    for (size_t i = 1; i + 1 < n; ++i) y[i] = a * x[i - 1] + b * x[i] + c * x[i + 1];
}

// z[i] = x[i*S]*y[i*S] with a compile-time stride S (lets gcc use shuffles
// or fixed-pattern gathers rather than a generic runtime-stride gather).
template <typename T, size_t S>
inline void mul_strided_c(T* __restrict z, const T* __restrict x, const T* __restrict y, size_t n) {
    for (size_t i = 0; i < n; ++i) z[i] = x[i * S] * y[i * S];
}

// z[i] = x[idx[i]] * y[idx[i]]  (true gather: data-dependent addresses; int32_t indices
// so gcc can map them to vgatherdps -- the first version used uint32_t widened
// to size_t and gcc reported "not vectorized: no vectype for stmt")
template <typename T>
inline void mul_gather(T* __restrict z, const T* __restrict x, const T* __restrict y,
                        const int32_t* __restrict idx, size_t n) {
    for (size_t i = 0; i < n; ++i) z[i] = x[idx[i]] * y[idx[i]];
}

// Work/traffic model per call, used for GFLOP/s, GiB/s and arithmetic
// intensity. "bytes" is the COMPULSORY traffic (each distinct element
// moved once, STREAM convention: write-allocate reads are not counted):
//   saxpy  : read x, read y, write y   -> 3 elems/iter,  2 flops
//   dot    : read x, read y            -> 2 elems/iter,  2 flops
//   mul    : read x, read y, write z   -> 3 elems/iter,  1 flop
//   stencil: read x, write y           -> 2 elems/iter,  5 flops (each x
//            element is re-read from L1 for the neighbouring outputs)
struct Work { double flops; double bytes; };
inline Work work(const std::string& k, size_t n, size_t elem_bytes) {
    double e = (double)elem_bytes, nn = (double)n;
    if (k == "saxpy")   return { 2.0 * nn,        3.0 * nn * e };
    if (k == "dot")     return { 2.0 * nn,        2.0 * nn * e };
    if (k == "mul")     return { 1.0 * nn,        3.0 * nn * e };
    if (k == "stencil") return { 5.0 * (nn - 2),  2.0 * nn * e };
    return { 0, 0 };
}

} // namespace kernels
