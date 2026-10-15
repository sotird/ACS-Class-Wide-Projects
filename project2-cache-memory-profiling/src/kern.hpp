// AVX-512 streaming kernels used by the bandwidth experiments. Written with
// intrinsics so the measurement is NOT at the mercy of auto-vectorization:
//  * loads are full 64 B lines (one cache line per load),
//  * every read kernel uses >= 4 independent integer accumulators (integer
//    adds are associative, so there is no loop-carried FP-add latency chain),
//  * stores are explicit (regular or non-temporal).
// All pointers must be 64 B aligned (VirtualAlloc returns 64 KiB-aligned).
#pragma once
#include "common.hpp"

namespace kern {

inline uint64_t reduce(__m512i a) {
    alignas(64) uint32_t t[16];
    _mm512_store_si512(t, a);
    uint64_t s = 0; for (int i = 0; i < 16; ++i) s += t[i];
    return s;
}

// Sequential read, bytes multiple of 512.
inline uint64_t read_seq(const char* p, size_t bytes) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    for (size_t o = 0; o < bytes; o += 512) {
        a0 = _mm512_add_epi32(a0, _mm512_load_si512(p + o));
        a1 = _mm512_add_epi32(a1, _mm512_load_si512(p + o + 64));
        a2 = _mm512_add_epi32(a2, _mm512_load_si512(p + o + 128));
        a3 = _mm512_add_epi32(a3, _mm512_load_si512(p + o + 192));
        a4 = _mm512_add_epi32(a4, _mm512_load_si512(p + o + 256));
        a5 = _mm512_add_epi32(a5, _mm512_load_si512(p + o + 320));
        a6 = _mm512_add_epi32(a6, _mm512_load_si512(p + o + 384));
        a7 = _mm512_add_epi32(a7, _mm512_load_si512(p + o + 448));
    }
    return reduce(_mm512_add_epi32(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)),
                                   _mm512_add_epi32(_mm512_add_epi32(a4, a5), _mm512_add_epi32(a6, a7))));
}

// One 64 B load every `stride` bytes over [0, span).
inline uint64_t read_stride(const char* p, size_t span, size_t stride) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    size_t o = 0;
    for (; o + 3 * stride < span; o += 4 * stride) {
        a0 = _mm512_add_epi32(a0, _mm512_load_si512(p + o));
        a1 = _mm512_add_epi32(a1, _mm512_load_si512(p + o + stride));
        a2 = _mm512_add_epi32(a2, _mm512_load_si512(p + o + 2 * stride));
        a3 = _mm512_add_epi32(a3, _mm512_load_si512(p + o + 3 * stride));
    }
    for (; o < span; o += stride) a0 = _mm512_add_epi32(a0, _mm512_load_si512(p + o));
    return reduce(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)));
}

// Independent loads of n lines at precomputed byte offsets (memory-level
// parallelism: addresses do not depend on loaded data).
inline uint64_t read_lines_at(const char* p, const uint32_t* offs, size_t n) {
    __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        a0 = _mm512_add_epi32(a0, _mm512_load_si512(p + offs[i]));
        a1 = _mm512_add_epi32(a1, _mm512_load_si512(p + offs[i + 1]));
        a2 = _mm512_add_epi32(a2, _mm512_load_si512(p + offs[i + 2]));
        a3 = _mm512_add_epi32(a3, _mm512_load_si512(p + offs[i + 3]));
    }
    for (; i < n; ++i) a0 = _mm512_add_epi32(a0, _mm512_load_si512(p + offs[i]));
    return reduce(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)));
}

// Sequential write (regular stores: each line is first read for ownership).
inline void write_seq(char* p, size_t bytes) {
    __m512i v = _mm512_set1_epi32(0x01010101);
    for (size_t o = 0; o < bytes; o += 512) {
        _mm512_store_si512(p + o, v);       _mm512_store_si512(p + o + 64, v);
        _mm512_store_si512(p + o + 128, v); _mm512_store_si512(p + o + 192, v);
        _mm512_store_si512(p + o + 256, v); _mm512_store_si512(p + o + 320, v);
        _mm512_store_si512(p + o + 384, v); _mm512_store_si512(p + o + 448, v);
    }
}
// Sequential write with non-temporal stores (bypass the cache: no read-for-ownership).
inline void write_seq_nt(char* p, size_t bytes) {
    __m512i v = _mm512_set1_epi32(0x01010101);
    for (size_t o = 0; o < bytes; o += 512) {
        _mm512_stream_si512((__m512i*)(p + o), v);       _mm512_stream_si512((__m512i*)(p + o + 64), v);
        _mm512_stream_si512((__m512i*)(p + o + 128), v); _mm512_stream_si512((__m512i*)(p + o + 192), v);
        _mm512_stream_si512((__m512i*)(p + o + 256), v); _mm512_stream_si512((__m512i*)(p + o + 320), v);
        _mm512_stream_si512((__m512i*)(p + o + 384), v); _mm512_stream_si512((__m512i*)(p + o + 448), v);
    }
    _mm_sfence();
}

// Mixed read/write stream: in every group of 10 consecutive lines, exactly R
// are read and 10-R are written, interleaved (not blocked) by the same
// Bresenham rule at compile time. bytes must be a multiple of 640.
constexpr bool is_read(int R, int k) { return ((k + 1) * R) / 10 > (k * R) / 10; }
template <int R, int K>
inline void mix_line(char* q, __m512i& acc, __m512i v) {
    if constexpr (is_read(R, K)) acc = _mm512_add_epi32(acc, _mm512_load_si512(q + 64 * K));
    else _mm512_store_si512(q + 64 * K, v);
}
template <int R>
inline uint64_t mix_seq(char* p, size_t bytes) {
    __m512i acc = _mm512_setzero_si512(), v = _mm512_set1_epi32(0x01010101);
    for (size_t o = 0; o < bytes; o += 640) {
        char* q = p + o;
        mix_line<R, 0>(q, acc, v); mix_line<R, 1>(q, acc, v); mix_line<R, 2>(q, acc, v);
        mix_line<R, 3>(q, acc, v); mix_line<R, 4>(q, acc, v); mix_line<R, 5>(q, acc, v);
        mix_line<R, 6>(q, acc, v); mix_line<R, 7>(q, acc, v); mix_line<R, 8>(q, acc, v);
        mix_line<R, 9>(q, acc, v);
    }
    return reduce(acc);
}

// xorshift64: independent random addresses without a stored index array
// (so index loads do not add their own memory traffic).
inline uint64_t xorshift(uint64_t& s) { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }

} // namespace kern
