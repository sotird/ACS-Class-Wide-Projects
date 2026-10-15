// Independent reference implementation of the QD=1 latency measurements and
// the "misleading setup" demonstrations, using only Win32 synchronous I/O and
// QueryPerformanceCounter -- no FIO. Used to (1) cross-validate FIO's QD=1
// numbers and (2) demonstrate pitfalls with a second tool.
//
// SAFETY: refuses to touch any path outside T:\ (the dedicated test
// partition); it only ever creates/overwrites ONE scratch file and
// never opens a volume or raw device.
//
// Modes (all single-threaded, queue depth 1, one outstanding synchronous I/O):
//   randread4k_direct     FILE_FLAG_NO_BUFFERING (bypasses the OS page cache)
//   randwrite4k_direct    NO_BUFFERING (device write cache may still absorb)
//   randwrite4k_wt        NO_BUFFERING | WRITE_THROUGH (forces media commit)
//   seqread128k_direct, seqwrite128k_direct
//   randread4k_buffered   normal cached I/O -> served from RAM after first touch
//   randwrite4k_buffered  normal cached I/O, no flush -> completes into RAM
//   vdl                   random 4 KiB direct reads of a file that was extended
//                         (SetEndOfFile) but NEVER WRITTEN, then the same file
//                         after it has been written: NTFS returns zeros for
//                         never-written ranges without touching the SSD.
//
// Usage: ref_qd1.exe <mode> <path> <seconds> <range_mib> <out_csv>
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

static double qpc_us(LONGLONG ticks, double freq) { return ticks * 1e6 / freq; }

struct Stats { size_t n; double secs, mean, p50, p90, p95, p99, p999, mx; };
static Stats summarize(std::vector<double>& us, double secs) {
    std::sort(us.begin(), us.end());
    Stats s{}; s.n = us.size(); s.secs = secs;
    double sum = 0; for (double v : us) sum += v;
    auto q = [&](double p) { return us[std::min(us.size() - 1, (size_t)(p * (us.size() - 1) + 0.5))]; };
    s.mean = sum / us.size(); s.p50 = q(0.5); s.p90 = q(0.9); s.p95 = q(0.95); s.p99 = q(0.99); s.p999 = q(0.999); s.mx = us.back();
    return s;
}

static bool path_ok(const std::string& p) {
    if (p.size() < 4) return false;
    char d = (char)toupper(p[0]);
    return d == 'T' && p[1] == ':' && (p[2] == '\\' || p[2] == '/');
}

// One timed run: returns per-op latencies in microseconds.
static std::vector<double> run(HANDLE h, bool write, bool random, DWORD bs, uint64_t range_bytes, double seconds, void* buf) {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    std::mt19937_64 rng(12345);
    uint64_t nblocks = range_bytes / bs;
    std::vector<double> lat; lat.reserve(1 << 20);
    LARGE_INTEGER t0, a, b; QueryPerformanceCounter(&t0);
    uint64_t seq = 0;
    for (;;) {
        uint64_t blk = random ? (rng() % nblocks) : (seq++ % nblocks);
        LARGE_INTEGER off; off.QuadPart = (LONGLONG)(blk * bs);
        OVERLAPPED ov{}; ov.Offset = off.LowPart; ov.OffsetHigh = (DWORD)off.HighPart;
        DWORD done = 0;
        QueryPerformanceCounter(&a);
        BOOL ok = write ? WriteFile(h, buf, bs, &done, &ov) : ReadFile(h, buf, bs, &done, &ov);
        QueryPerformanceCounter(&b);
        if (!ok || done != bs) { std::fprintf(stderr, "I/O failed: err=%lu\n", GetLastError()); std::exit(4); }
        lat.push_back(qpc_us(b.QuadPart - a.QuadPart, (double)f.QuadPart));
        if ((double)(b.QuadPart - t0.QuadPart) / f.QuadPart >= seconds) break;
    }
    return lat;
}

static HANDLE open_file(const std::string& path, DWORD access, DWORD flags, DWORD disp) {
    HANDLE h = CreateFileA(path.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, disp, flags, nullptr);
    if (h == INVALID_HANDLE_VALUE) { std::fprintf(stderr, "CreateFile failed (%lu) for %s\n", GetLastError(), path.c_str()); std::exit(5); }
    return h;
}

int main(int argc, char** argv) {
    if (argc != 6) { std::fprintf(stderr, "usage: %s <mode> <path> <seconds> <range_mib> <out_csv>\n", argv[0]); return 1; }
    std::string mode = argv[1], path = argv[2]; double seconds = std::atof(argv[3]);
    uint64_t range = (uint64_t)std::atoll(argv[4]) << 20;
    if (!path_ok(path)) { std::fprintf(stderr, "REFUSING: path must be on T:\\ (the dedicated test partition)\n"); return 3; }
    if (range > (1536ull << 20)) { std::fprintf(stderr, "REFUSING: range too large for the 1.6 GB test partition\n"); return 3; }

    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 4);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    void* buf = VirtualAlloc(nullptr, 1 << 20, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
    std::mt19937_64 g(777); for (size_t i = 0; i < (1u << 20) / 8; ++i) ((uint64_t*)buf)[i] = g();   // incompressible payload

    std::ofstream csv(argv[5], std::ios::app);
    if (csv.tellp() == 0) csv << "mode,ops,seconds,iops,mean_us,p50_us,p90_us,p95_us,p99_us,p999_us,max_us\n";
    auto emit = [&](const std::string& name, std::vector<double>& lat, double secs) {
        Stats s = summarize(lat, secs);
        csv << name << "," << s.n << "," << s.secs << "," << s.n / s.secs << "," << s.mean << "," << s.p50 << "," << s.p90 << "," << s.p95
            << "," << s.p99 << "," << s.p999 << "," << s.mx << "\n";
        std::printf("%-22s ops=%zu iops=%.0f mean=%.1fus p50=%.1f p95=%.1f p99=%.1f p99.9=%.1f max=%.1f\n",
                    name.c_str(), s.n, s.n / s.secs, s.mean, s.p50, s.p95, s.p99, s.p999, s.mx);
    };

    // Common case: file already exists (created/preconditioned by run_fio.py).
    auto timed = [&](bool write, bool random, DWORD bs, DWORD flags, const char* label) {
        HANDLE h = open_file(path, GENERIC_READ | GENERIC_WRITE, flags, OPEN_EXISTING);
        LARGE_INTEGER sz; GetFileSizeEx(h, &sz);
        if ((uint64_t)sz.QuadPart < range) { std::fprintf(stderr, "file smaller than requested range\n"); std::exit(6); }
        run(h, write, random, bs, range, 0.3, buf);                      // warm-up
        auto lat = run(h, write, random, bs, range, seconds, buf);
        CloseHandle(h);
        emit(label, lat, seconds);
    };
    const DWORD DIRECT = FILE_FLAG_NO_BUFFERING;
    if (mode == "randread4k_direct")       timed(false, true, 4096, DIRECT, mode.c_str());
    else if (mode == "randwrite4k_direct") timed(true, true, 4096, DIRECT, mode.c_str());
    else if (mode == "randwrite4k_wt")     timed(true, true, 4096, DIRECT | FILE_FLAG_WRITE_THROUGH, mode.c_str());
    else if (mode == "seqread128k_direct") timed(false, false, 131072, DIRECT, mode.c_str());
    else if (mode == "seqwrite128k_direct")timed(true, false, 131072, DIRECT, mode.c_str());
    else if (mode == "randread4k_buffered")  timed(false, true, 4096, 0, mode.c_str());
    else if (mode == "randwrite4k_buffered") timed(true, true, 4096, 0, mode.c_str());
    else if (mode == "vdl") {
        // 1) extend without writing -> valid-data-length stays 0
        HANDLE h = open_file(path, GENERIC_READ | GENERIC_WRITE, 0, CREATE_ALWAYS);
        LARGE_INTEGER sz; sz.QuadPart = (LONGLONG)range;
        SetFilePointerEx(h, sz, nullptr, FILE_BEGIN); SetEndOfFile(h); CloseHandle(h);
        h = open_file(path, GENERIC_READ, DIRECT, OPEN_EXISTING);
        run(h, false, true, 4096, range, 0.3, buf);
        auto lat = run(h, false, true, 4096, range, seconds, buf);
        CloseHandle(h);
        emit("vdl_unwritten_file", lat, seconds);
        // 2) write it for real (direct, sequential 1 MiB), then read again
        h = open_file(path, GENERIC_READ | GENERIC_WRITE, DIRECT, OPEN_EXISTING);
        for (uint64_t off = 0; off < range; off += (1 << 20)) {
            OVERLAPPED ov{}; ov.Offset = (DWORD)off; ov.OffsetHigh = (DWORD)(off >> 32); DWORD done;
            WriteFile(h, buf, 1 << 20, &done, &ov);
        }
        FlushFileBuffers(h); CloseHandle(h);
        h = open_file(path, GENERIC_READ, DIRECT, OPEN_EXISTING);
        run(h, false, true, 4096, range, 0.3, buf);
        lat = run(h, false, true, 4096, range, seconds, buf);
        CloseHandle(h);
        emit("vdl_written_file", lat, seconds);
    } else { std::fprintf(stderr, "unknown mode %s\n", mode.c_str()); return 1; }
    return 0;
}
