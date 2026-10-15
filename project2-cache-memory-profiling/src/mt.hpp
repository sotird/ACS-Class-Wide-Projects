// Multi-threaded load generator + latency probe (the MLC-style "loaded
// latency" method): N pinned generator threads stream over private 512 MiB
// regions at a controllable intensity while ONE separately-pinned probe
// thread runs a dependent pointer-chase on its own buffer and reports the
// latency it sees under that load. Throughput = bytes the generators moved
// per second; latency = what the probe measured at the same time.
//
// Intensity knobs: number of active generator threads, and an injected delay
// (rdtsc spin) after every 40 KiB chunk.
// Only ONE hardware thread per physical core is used (even logical CPU
// numbers); SMT siblings are left idle.
#pragma once
#include "common.hpp"
#include "chase.hpp"
#include "kern.hpp"

namespace bench {

enum Mode { M_READ = 0, M_WRITE = 1, M_WRITE_NT = 2, M_MIX70 = 3, M_MIX50 = 4,
            M_STRIDE = 5, M_RANDLINE = 6, M_RANDBLOCK = 7 };

inline const char* mode_name(int m) {
    static const char* n[] = {"100R", "100W", "100W_NT", "70R30W", "50R50W", "stride", "randline", "randblock"};
    return n[m];
}

struct alignas(64) PaddedCounter { std::atomic<uint64_t> bytes{0}; char pad[56]; };

class LoadGen {
public:
    static constexpr size_t CHUNK = 40960;   // = 80 x 512 B = 64 x 640 B

    LoadGen(int max_threads, std::vector<int> cpus, size_t region_bytes)
        : max_threads_(max_threads), cpus_(std::move(cpus)), region_(region_bytes), counters_(max_threads) {
        for (int t = 0; t < max_threads_; ++t) threads_.emplace_back([this, t] { worker(t); });
        while (ready_.load() < max_threads_) Sleep(5);
    }
    ~LoadGen() { quit_ = true; for (auto& th : threads_) th.join(); }

    void configure(int active, int mode, int param, double delay_us) {
        active_ = active; mode_ = mode; param_ = param;
        delay_ticks_ = (uint64_t)(delay_us * 1e-6 * tsc_ghz_ * 1e9);
    }
    uint64_t total_bytes() const {
        uint64_t s = 0; for (auto& c : counters_) s += c.bytes.load(std::memory_order_relaxed); return s;
    }
    static double tsc_ghz_;

private:
    void worker(int tid) {
        pin_thread(cpus_[tid]);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
        char* region = alloc_pages(region_, true);          // first-touched by (and local to) this thread
        std::vector<uint32_t> tab; int tab_mode = -1, tab_param = -1; size_t tpos = 0;
        constexpr size_t TAB_N = 640 * 1638;                 // ~1 M entries, a multiple of 640
        size_t pos = 0; uint64_t local = 0, sink = 0;
        ready_++;
        while (!quit_) {
            if (tid >= active_.load(std::memory_order_relaxed)) { Sleep(1); continue; }
            int m = mode_.load(std::memory_order_relaxed), prm = param_.load(std::memory_order_relaxed);
            if (pos + CHUNK > region_) pos = 0;
            char* p = region + pos;
            uint64_t counted = CHUNK;
            switch (m) {
            case M_READ:     sink += kern::read_seq(p, CHUNK); pos += CHUNK; break;
            case M_WRITE:    kern::write_seq(p, CHUNK); pos += CHUNK; break;
            case M_WRITE_NT: kern::write_seq_nt(p, CHUNK); pos += CHUNK; break;
            case M_MIX70:    sink += kern::mix_seq<7>(p, CHUNK); pos += CHUNK; break;
            case M_MIX50:    sink += kern::mix_seq<5>(p, CHUNK); pos += CHUNK; break;
            case M_STRIDE:   sink += kern::read_stride(p, CHUNK, (size_t)prm); counted = (CHUNK / prm) * 64; pos += CHUNK; break;
            case M_RANDLINE: case M_RANDBLOCK: {
                // Random addresses come from a PRECOMPUTED per-thread table (1 M entries =
                // 4 MiB). Generating them inside the timed loop (inline PRNG) halves a
                // core's memory-level parallelism -- see exp8 pitfall (d).
                const size_t unit = (m == M_RANDLINE) ? 64 : (size_t)prm;
                if (tab_mode != m || tab_param != prm) {
                    tab.resize(TAB_N); std::mt19937_64 g(0xABCDEFu + tid);
                    const uint64_t nunits = region_ / unit;
                    for (auto& e : tab) e = (uint32_t)((g() % nunits) * unit);
                    tab_mode = m; tab_param = prm; tpos = 0;
                }
                const size_t per_chunk = CHUNK / unit;                // 640 lines, or CHUNK/B blocks
                if (tpos + per_chunk > TAB_N) tpos = 0;
                const uint32_t* t = tab.data() + tpos; tpos += per_chunk;
                __m512i a0 = _mm512_setzero_si512(), a1 = a0, a2 = a0, a3 = a0;
                if (m == M_RANDLINE) {
                    for (size_t i = 0; i < per_chunk; i += 4) {
                        a0 = _mm512_add_epi32(a0, _mm512_load_si512(region + t[i]));
                        a1 = _mm512_add_epi32(a1, _mm512_load_si512(region + t[i + 1]));
                        a2 = _mm512_add_epi32(a2, _mm512_load_si512(region + t[i + 2]));
                        a3 = _mm512_add_epi32(a3, _mm512_load_si512(region + t[i + 3]));
                    }
                } else {
                    for (size_t i = 0; i < per_chunk; ++i) {
                        const char* q = region + t[i];
                        for (size_t o = 0; o < unit; o += 128) {
                            a0 = _mm512_add_epi32(a0, _mm512_load_si512(q + o));
                            a1 = _mm512_add_epi32(a1, _mm512_load_si512(q + o + 64));
                        }
                    }
                }
                sink += kern::reduce(_mm512_add_epi32(_mm512_add_epi32(a0, a1), _mm512_add_epi32(a2, a3)));
                break;
            }
            }
            local += counted;
            counters_[tid].bytes.store(local, std::memory_order_relaxed);
            uint64_t d = delay_ticks_.load(std::memory_order_relaxed);
            if (d) spin_ticks(d);
        }
        volatile uint64_t s2 = sink; (void)s2;
        free_pages(region);
    }

    int max_threads_;
    std::vector<int> cpus_;
    size_t region_;
    std::vector<PaddedCounter> counters_;
    std::vector<std::thread> threads_;
    std::atomic<int> ready_{0}, active_{0}, mode_{0}, param_{64};
    std::atomic<uint64_t> delay_ticks_{0};
    std::atomic<bool> quit_{false};
};
inline double LoadGen::tsc_ghz_ = 0;

struct ProbeStats { double mean_ns = 0, p50_ns = 0, p5_ns = 0, p95_ns = 0; long hops = 0; };

// Runs the dependent chase for `seconds`, in batches of `batch` hops (each
// batch is >> timer resolution); statistics are over the batch means.
inline ProbeStats probe_window(const Chase& c, uint32_t& off, double seconds, int batch = 2048) {
    std::vector<double> b;
    auto t_end = Clock::now() + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(seconds));
    while (Clock::now() < t_end) {
        auto t0 = Clock::now();
        off = chase_run(c.base, off, batch);
        b.push_back(seconds_since(t0) * 1e9 / batch);
    }
    ProbeStats s;
    s.hops = (long)b.size() * batch;
    s.mean_ns = std::accumulate(b.begin(), b.end(), 0.0) / b.size();
    s.p50_ns = median(b); s.p5_ns = quantile(b, 0.05); s.p95_ns = quantile(b, 0.95);
    return s;
}

struct LoadedPoint { double gbps = 0; ProbeStats lat; };

// One loaded-latency measurement: configure generators, let them reach steady
// state, then measure bytes moved and probe latency over the same window.
inline LoadedPoint measure_loaded(LoadGen& g, const Chase& probe, uint32_t& off, int active, int mode, int param,
                                  double delay_us, double warm_s = 0.30, double window_s = 0.60) {
    g.configure(active, mode, param, delay_us);
    Sleep((DWORD)(warm_s * 1000));
    uint64_t b0 = g.total_bytes(); auto t0 = Clock::now();
    LoadedPoint p;
    p.lat = probe_window(probe, off, window_s);
    uint64_t b1 = g.total_bytes(); double dt = seconds_since(t0);
    p.gbps = (b1 - b0) / dt / 1e9;
    g.configure(0, mode, param, 0);
    return p;
}

// Generator CPUs: one logical CPU per physical core (even numbers 0,2,...),
// leaving the highest cores for the probe (CPU 28) and the controller (CPU 30).
inline std::vector<int> generator_cpus(int n) {
    std::vector<int> v; for (int i = 0; i < n; ++i) v.push_back(2 * i); return v;
}
constexpr int PROBE_CPU = 28;
constexpr int CONTROL_CPU = 30;
constexpr int MAX_GEN_THREADS = 14;       // cores 0..13

} // namespace bench
