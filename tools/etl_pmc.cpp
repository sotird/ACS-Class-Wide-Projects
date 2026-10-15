// etl_pmc: read hardware performance-counter values from a WPR/ETW trace (.etl) recorded with
//   <HardwareCounter><Counters>...</Counters><Events><Event Value="CSwitch"/></Events></HardwareCounter>
// and print, per process id, the counter totals attributed to its threads.  No Windows Performance
// Toolkit needed: it uses the documented trace-consumer API (OpenTrace / ProcessTrace).
//
// How the numbers are obtained: every CSwitch (context-switch) event carries an extended-data item
// EVENT_HEADER_EXT_TYPE_PMC_COUNTERS (0x000F) with the raw, free-running counter values of the logical CPU
// at the switch.  The delta between two consecutive switches on the same CPU is the work done by the
// thread that ran in between (= the NewThread of the first switch).  Thread->process comes from the
// Thread Start / DCStart events.
//
// Usage: etl_pmc.exe trace.etl [--dump N]      (--dump prints the first N CSwitch records raw)
// Build: g++ -O2 -std=c++17 -static -o etl_pmc.exe etl_pmc.cpp -ladvapi32      (MinGW-w64)
#define INITGUID
#include <windows.h>
#include <evntrace.h>
#include <evntcons.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <vector>
#include <string>
#include <cstdint>

#ifndef EVENT_HEADER_EXT_TYPE_PMC_COUNTERS
#define EVENT_HEADER_EXT_TYPE_PMC_COUNTERS 0x000F
#endif

// {3D6FA8D1-FE05-11D0-9DDA-00C04FD7BA7C}  NT Kernel "Thread" event class
static const GUID kThreadGuid = {0x3d6fa8d1, 0xfe05, 0x11d0, {0x9d, 0xda, 0x00, 0xc0, 0x4f, 0xd7, 0xba, 0x7c}};

static std::map<uint32_t, uint32_t> g_tid2pid;                 // thread id -> process id
struct CpuState { bool have = false; std::vector<uint64_t> last; uint32_t running_tid = 0; };
static std::map<int, CpuState> g_cpu;
static std::map<uint32_t, std::vector<uint64_t>> g_pid_total;  // pid -> counter totals
static std::map<uint32_t, uint64_t> g_pid_switches;
static size_t g_ncnt = 0;
static long g_dump = 0, g_cswitch = 0, g_with_pmc = 0;
static uint64_t g_first_ts = 0, g_last_ts = 0;

static void WINAPI on_event(PEVENT_RECORD rec) {
    const EVENT_HEADER& h = rec->EventHeader;
    if (!IsEqualGUID(h.ProviderId, kThreadGuid)) return;
    const uint8_t op = h.EventDescriptor.Opcode;
    const uint8_t* ud = (const uint8_t*)rec->UserData;
    if ((op == 1 || op == 3) && rec->UserDataLength >= 8) {           // Thread Start / DCStart: ProcessId, TThreadId
        uint32_t pid = *(const uint32_t*)ud, tid = *(const uint32_t*)(ud + 4);
        g_tid2pid[tid] = pid;
        return;
    }
    if (op != 36 || rec->UserDataLength < 8) return;                   // CSwitch
    ++g_cswitch;
    uint32_t newtid = *(const uint32_t*)ud;
    const uint64_t* ctr = nullptr; size_t n = 0;
    for (USHORT i = 0; i < rec->ExtendedDataCount; ++i) {
        const EVENT_HEADER_EXTENDED_DATA_ITEM& it = rec->ExtendedData[i];
        if (it.ExtType == EVENT_HEADER_EXT_TYPE_PMC_COUNTERS) { ctr = (const uint64_t*)(uintptr_t)it.DataPtr; n = it.DataSize / 8; }
    }
    if (g_dump > 0 && n > 0) {
        --g_dump;
        std::printf("cpu=%u ts=%llu new_tid=%u old_tid=%u ext_items=%u pmc_n=%zu :", rec->BufferContext.ProcessorNumber, (unsigned long long)h.TimeStamp.QuadPart,
                    newtid, *(const uint32_t*)(ud + 4), rec->ExtendedDataCount, n);
        for (size_t k = 0; k < n; ++k) std::printf(" %llu", (unsigned long long)ctr[k]);
        std::printf("\n");
    }
    if (!ctr || n == 0) return;
    ++g_with_pmc; g_ncnt = n;
    if (!g_first_ts) g_first_ts = h.TimeStamp.QuadPart; g_last_ts = h.TimeStamp.QuadPart;
    CpuState& st = g_cpu[rec->BufferContext.ProcessorNumber];
    if (st.have && st.running_tid) {
        auto pit = g_tid2pid.find(st.running_tid);
        uint32_t pid = pit == g_tid2pid.end() ? 0xFFFFFFFFu : pit->second;
        auto& tot = g_pid_total[pid]; if (tot.size() < n) tot.assign(n, 0);
        for (size_t k = 0; k < n && k < st.last.size(); ++k) tot[k] += ctr[k] - st.last[k];
        ++g_pid_switches[pid];
    }
    st.last.assign(ctr, ctr + n); st.have = true; st.running_tid = newtid;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s trace.etl [--dump N]\n", argv[0]); return 1; }
    for (int i = 2; i + 1 < argc; ++i) if (!std::strcmp(argv[i], "--dump")) g_dump = std::atol(argv[i + 1]);
    wchar_t wpath[1024]; MultiByteToWideChar(CP_ACP, 0, argv[1], -1, wpath, 1024);
    EVENT_TRACE_LOGFILEW lf; std::memset(&lf, 0, sizeof lf);
    lf.LogFileName = wpath;
    lf.ProcessTraceMode = PROCESS_TRACE_MODE_EVENT_RECORD | PROCESS_TRACE_MODE_RAW_TIMESTAMP;
    lf.EventRecordCallback = on_event;
    TRACEHANDLE th = OpenTraceW(&lf);
    if (th == INVALID_PROCESSTRACE_HANDLE) { std::fprintf(stderr, "OpenTrace failed: %lu\n", GetLastError()); return 2; }
    ULONG rc = ProcessTrace(&th, 1, nullptr, nullptr);
    CloseTrace(th);
    if (rc != ERROR_SUCCESS) { std::fprintf(stderr, "ProcessTrace failed: %lu\n", rc); return 3; }
    std::fprintf(stderr, "CSwitch events: %ld, with PMC data: %ld, counters per event: %zu, threads known: %zu\n", g_cswitch, g_with_pmc, g_ncnt, g_tid2pid.size());
    std::printf("pid,switches");
    for (size_t k = 0; k < g_ncnt; ++k) std::printf(",counter%zu", k);
    std::printf("\n");
    for (auto& kv : g_pid_total) {
        std::printf("%u,%llu", kv.first, (unsigned long long)g_pid_switches[kv.first]);
        for (size_t k = 0; k < g_ncnt; ++k) std::printf(",%llu", (unsigned long long)(k < kv.second.size() ? kv.second[k] : 0));
        std::printf("\n");
    }
    return 0;
}
