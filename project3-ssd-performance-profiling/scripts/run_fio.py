#!/usr/bin/env python3
"""Orchestrates every Project 3 FIO experiment against ONE scratch file on the
dedicated T: test partition.

Safety (data-loss warning in the assignment):
  * refuses to run unless T: is the small NTFS volume labelled "Testing" (1-2 GiB), is the LAST
    partition of the disk, and contains nothing except our own scratch directory (System Volume
    Information excepted). The partition shares a physical SSD with a data volume (G:), which is
    never opened: FIO only ever touches T:\\ssdbench\\fio.dat.
  * file-level I/O only (no raw \\\\.\\PhysicalDrive / volume handles).
Method:
  * direct I/O (FILE_FLAG_NO_BUFFERING via direct=1), windowsaio, one 1500 MiB scratch file fully
    written (preconditioned) twice before any read test; payload incompressible and refilled for every I/O.
  * every job is saved as a .fio file in data/raw/jobs; raw FIO JSON is saved in data/raw/json; wall-clock
    start/end of every job in data/raw/job_times.csv (to join with the SMART temperature log).
    Re-running resumes: finished jobs (non-empty JSON) are skipped.
  * trials are INTERLEAVED (trial-major order) so slow drift of the device / temperature cannot bias one
    configuration; fresh-state experiments run first, destructive steady-state (heavy write) experiments last.
  * SMART temperature / latency-max counters (administrator rights required) are read by a persistent PowerShell
    helper after every job and every 30 s during runs >= 120 s, and logged with the job name in
    data/raw/logs/smart_log.csv. NOTE: a SMART query is a non-queued ATA command; polling every 2 s was measured to
    stall the drive for ~0.5 s in ~1/3 of the polls, which is why short jobs are never polled while running.

Usage: python run_fio.py [--stages 1,2,3,...] [--dry-run]
"""
import argparse
import atexit
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
JOBS, JSON_DIR, LOGS = (os.path.join(RAW, d) for d in ("jobs", "json", "logs"))
for d in (JOBS, JSON_DIR, LOGS):
    os.makedirs(d, exist_ok=True)

FIO = next((p for p in (os.path.expandvars(r"%LOCALAPPDATA%\Programs\fio-extracted\fio\fio.exe"),   # user-level extraction of the official MSI
                        r"C:\Program Files\fio\fio.exe") if os.path.exists(p)), "fio")
TARGET_DIR = r"T:\ssdbench"
FILE_PY = TARGET_DIR + r"\fio.dat"
FILE_FIO = r"T\:\ssdbench\fio.dat"          # FIO needs the drive-letter colon escaped
FILE_MIB = 1500                              # scratch file; T: is ~1.62 GiB
DEFAULT_SIZE = "1024m"                       # LBA range used unless an experiment says otherwise
PCTL = "50:75:90:95:99:99.5:99.9:99.99:99.999"
SCRIPTS = os.path.dirname(os.path.abspath(__file__))


def ps(cmd):
    return subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True).stdout.strip()


def safety_check():
    info = json.loads(ps("Get-Volume -DriveLetter T | Select-Object FileSystemLabel,FileSystem,Size,SizeRemaining | ConvertTo-Json"))
    assert info["FileSystemLabel"] == "Testing", f"T: label is {info['FileSystemLabel']!r}, expected 'Testing' -- refusing to run"
    assert info["FileSystem"] == "NTFS"
    assert info["Size"] <= 2 * 2**30, f"T: is {info['Size']/2**20:.0f} MiB -- too big to be the dedicated test partition; refusing"
    assert info["Size"] >= 1 * 2**30, f"T: is only {info['Size']/2**20:.0f} MiB; expected the ~1.6 GiB test partition"
    part = json.loads(ps("$p = Get-Partition -DriveLetter T; $last = (Get-Partition -DiskNumber $p.DiskNumber | Sort-Object Offset | Select-Object -Last 1).PartitionNumber; "
                         "[pscustomobject]@{Disk=$p.DiskNumber; Offset=$p.Offset; IsLast=($p.PartitionNumber -eq $last)} | ConvertTo-Json"))
    assert part["IsLast"], "T: is not the last partition of its disk -- refusing"
    entries = [e for e in os.listdir("T:\\") if e not in ("System Volume Information", "$RECYCLE.BIN", "ssdbench")]
    assert not entries, f"T: contains unexpected user data {entries}; refusing to run"
    os.makedirs(TARGET_DIR, exist_ok=True)
    others = [e for e in os.listdir(TARGET_DIR) if e not in ("fio.dat", "vdl.dat")]
    assert not others, f"unexpected files in {TARGET_DIR}: {others}"
    have = os.path.getsize(FILE_PY) if os.path.exists(FILE_PY) else 0
    need = max(0, FILE_MIB * 2**20 - have) + 80 * 2**20     # keep 80 MiB for the vdl pitfall file
    assert need < info["SizeRemaining"] + (0 if not os.path.exists(os.path.join(TARGET_DIR, "vdl.dat")) else os.path.getsize(os.path.join(TARGET_DIR, "vdl.dat"))), "not enough free space on T:"
    print(f"safety check OK: T: 'Testing' NTFS {info['Size']/2**20:.0f} MiB (disk {part['Disk']}, offset {part['Offset']}), {info['SizeRemaining']/2**20:.0f} MiB free, no user data")
    return info


class SmartProbe:
    """Persistent PowerShell process answering one SMART/reliability-counter query per line on stdin (~50 ms each)."""
    SCRIPT = ("$pd = Get-PhysicalDisk -DeviceNumber ((Get-Partition -DriveLetter T).DiskNumber); "
              "while (($l = [Console]::In.ReadLine()) -ne $null) { $rc = $pd | Get-StorageReliabilityCounter; "
              "$t = [DateTimeOffset]::UtcNow.ToUnixTimeMilliseconds() / 1000.0; "
              "'{0:F3},{1},{2},{3},{4},{5},{6}' -f $t, $rc.Temperature, $rc.Wear, $rc.PowerOnHours, $rc.ReadLatencyMax, $rc.WriteLatencyMax, $rc.FlushLatencyMax }")

    def __init__(self):
        self.lock = threading.Lock()
        self.path = os.path.join(LOGS, "smart_log.csv")
        self.p = subprocess.Popen(["powershell", "-NoProfile", "-NonInteractive", "-Command", self.SCRIPT], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)
        atexit.register(self.p.terminate)
        if not os.path.exists(self.path):
            open(self.path, "w").write("tag,epoch_s,temperature_c,wear,power_on_hours,read_latency_max_ms,write_latency_max_ms,flush_latency_max_ms\n")

    def query(self, tag):
        with self.lock:
            self.p.stdin.write("q\n"); self.p.stdin.flush()
            line = self.p.stdout.readline().strip()
            open(self.path, "a").write(f"{tag},{line}\n")
            return line


SMART = None


@dataclass
class Job:
    stage: str
    name: str
    rw: str
    bs: str
    iodepth: int = 1
    numjobs: int = 1
    mix: int = None
    size: str = DEFAULT_SIZE
    runtime: int = 8
    ramp: int = 2
    direct: int = 1
    ioengine: str = "windowsaio"
    compress: int = 0
    trials: int = 3
    logs: bool = False
    prewarm: bool = False     # read the first `size` of the file through the page cache before the job (buffered cache-warm case)
    extra: dict = field(default_factory=dict)


def write_jobfile(job, trial, path):
    L = ["; generated by run_fio.py (Project 3) -- do not edit", f"; stage={job.stage} trial={trial}",
         "[global]", f"ioengine={job.ioengine}", "thread", f"direct={job.direct}", f"filename={FILE_FIO}", f"size={job.size}",
         "time_based", f"runtime={job.runtime}", f"ramp_time={job.ramp}", "refill_buffers",
         f"buffer_compress_percentage={job.compress}", "norandommap", f"randseed={trial * 1000 + 7}", "group_reporting",
         "lat_percentiles=1", f"percentile_list={PCTL}", f"[{job.name}]", f"rw={job.rw}", f"bs={job.bs}",
         f"iodepth={job.iodepth}", f"numjobs={job.numjobs}"]
    if job.mix is not None:
        L.append(f"rwmixread={job.mix}")
    if job.logs:
        L += [f"write_bw_log=logs/{job.stage}__{job.name}__t{trial}", f"write_lat_log=logs/{job.stage}__{job.name}__t{trial}",
              f"write_iops_log=logs/{job.stage}__{job.name}__t{trial}", "log_avg_msec=1000"]
    for k, v in job.extra.items():
        L.append(f"{k}={v}")
    open(path, "w").write("\n".join(L) + "\n")


def run_job(job, trial, dry=False, monitor=False):
    base = f"{job.stage}__{job.name}__t{trial}"
    jf, out = os.path.join(JOBS, base + ".fio"), os.path.join(JSON_DIR, base + ".json")
    if os.path.exists(out) and os.path.getsize(out) > 0:   # fio creates the (empty) output file at start: only a finished run counts
        return
    write_jobfile(job, trial, jf)
    if dry:
        print("  [dry]", base); return
    mon = None
    if monitor:     # OS disk + per-CPU interrupt/DPC counters, 1 s resolution, for the duration of the job
        mon = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", os.path.join(SCRIPTS, "monitor_os.ps1"),
                                "-Seconds", str(job.runtime + job.ramp + 4), "-Out", os.path.join(LOGS, f"{base}_osmon.csv")],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.5)
    if job.prewarm:
        pw = os.path.join(JOBS, base + "_prewarm.fio")
        open(pw, "w").write("\n".join(["[prewarm]", f"filename={FILE_FIO}", "ioengine=sync", "direct=0", "rw=read", "bs=1m", f"size={job.size}", "thread", ""]))
        subprocess.run([FIO, pw, "--output-format=json", f"--output={os.path.join(LOGS, base + '_prewarm.json')}"], cwd=RAW, capture_output=True, text=True)
    stop = threading.Event()
    poller = None
    if SMART is not None and job.runtime >= 120:       # long sustained runs: temperature every 30 s (see module docstring)
        def poll():
            while not stop.wait(30):
                SMART.query(f"during:{base}")
        poller = threading.Thread(target=poll, daemon=True); poller.start()
    t0 = time.time()
    r = subprocess.run([FIO, jf, "--output-format=json", f"--output={out}", "--eta=never"], cwd=RAW, capture_output=True, text=True)
    t1 = time.time()
    stop.set()
    if poller is not None:
        poller.join()
    if SMART is not None:
        SMART.query(f"job_end:{base}")
    if mon is not None:
        mon.wait()
    if r.returncode != 0 or not os.path.exists(out) or os.path.getsize(out) == 0:
        print(f"FIO FAILED for {base}: rc={r.returncode}\n{r.stdout}\n{r.stderr}")
        sys.exit(2)
    with open(os.path.join(RAW, "job_times.csv"), "a") as f:
        if f.tell() == 0:
            f.write("job,start_epoch_s,end_epoch_s\n")
        f.write(f"{base},{t0:.3f},{t1:.3f}\n")
    print(f"  {base}  ({t1 - t0:.0f}s)", flush=True)
    time.sleep(2)


def run_stage(jobs, dry=False, monitor=False):
    for t in range(max(j.trials for j in jobs)):
        for j in jobs:
            if t < j.trials:
                run_job(j, t, dry, monitor)


def precondition(dry=False):
    path = os.path.join(JOBS, "precondition.fio")
    open(path, "w").write(f"""; sequential preconditioning: write the WHOLE {FILE_MIB} MiB file twice with random incompressible data
[global]
ioengine=windowsaio
thread
direct=1
filename={FILE_FIO}
size={FILE_MIB}m
refill_buffers
buffer_compress_percentage=0
group_reporting
[precondition]
rw=write
bs=1m
iodepth=8
loops=2
""")
    marker = os.path.join(RAW, "precondition.done")
    if os.path.exists(marker) and os.path.exists(FILE_PY):
        print("precondition already done"); return
    if dry:
        print("  [dry] precondition"); return
    r = subprocess.run([FIO, path, "--output-format=json", f"--output={os.path.join(JSON_DIR, 'precondition.json')}"], cwd=RAW, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout, r.stderr); sys.exit(2)
    assert os.path.getsize(FILE_PY) >= FILE_MIB * 2**20, "scratch file smaller than expected"
    open(marker, "w").write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print("precondition done:", os.path.getsize(FILE_PY) / 2**20, "MiB written twice")


# ----------------------------------------------------------------- stages ----
def stage1():   # zero-queue baselines (QD=1), fresh state
    J = []
    for nm, rw, bs in (("randread4k", "randread", "4k"), ("randwrite4k", "randwrite", "4k"), ("seqread128k", "read", "128k"), ("seqwrite128k", "write", "128k")):
        J.append(Job("s1_zq", f"{nm}_qd1", rw, bs, 1, runtime=20, ramp=2, trials=5))
    return J


def stage2():   # block-size x pattern x direction, QD1 (latency) and QD32 (throughput)
    J = []
    for pat, sizes in (("rand", ["4k", "16k", "32k", "64k", "128k", "256k"]), ("seq", ["4k", "16k", "32k", "64k", "128k", "256k", "512k", "1m"])):
        for d in ("read", "write"):
            rw = {"randread": "randread", "randwrite": "randwrite", "seqread": "read", "seqwrite": "write"}[pat + d]
            for bs in sizes:
                for qd in (1, 32):
                    J.append(Job("s2_bs", f"{pat}{d}_bs{bs}_qd{qd}", rw, bs, qd, runtime=6, ramp=2, trials=3))
    return J


def stage3():   # read/write mix at fixed block size and pattern
    J = []
    for mix, label in ((100, "100R"), (0, "100W"), (70, "70R30W"), (50, "50R50W")):
        for qd in (1, 32):
            J.append(Job("s3_mix", f"rand4k_{label}_qd{qd}", "randrw", "4k", qd, mix=mix, runtime=10, ramp=2, trials=3))
        J.append(Job("s3_mix", f"seq128k_{label}_qd4", "rw", "128k", 4, mix=mix, runtime=10, ramp=2, trials=3))
    return J


def stage4():   # queue-depth / parallelism sweep
    J = []
    for nm, rw, bs, mix in (("randread4k", "randread", "4k", None), ("randwrite4k", "randwrite", "4k", None), ("randmix70_4k", "randrw", "4k", 70)):
        for qd in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            J.append(Job("s4_qd", f"{nm}_qd{qd}", rw, bs, qd, mix=mix, runtime=8, ramp=2, trials=3))
    for nm, rw in (("seqread128k", "read"), ("seqwrite128k", "write")):
        for qd in (1, 2, 4, 8, 16, 32, 64):
            J.append(Job("s4_qd", f"{nm}_qd{qd}", rw, "128k", qd, runtime=8, ramp=2, trials=3))
    for nj in (1, 2, 4, 8, 16, 32):
        J.append(Job("s4_nj", f"randread4k_qd1_nj{nj}", "randread", "4k", 1, numjobs=nj, runtime=8, ramp=2, trials=3))
    return J


def stage5():   # tail latency, long runs
    J = []
    for nm, rw in (("randread4k", "randread"), ("randwrite4k", "randwrite")):
        for qd in (1, 4, 16, 32, 128):
            J.append(Job("s5_tail", f"{nm}_qd{qd}", rw, "4k", qd, runtime=45, ramp=3, trials=2))
    return J


RANGES = ("1m", "4m", "16m", "64m", "256m", "1024m", "1500m")


def stage6():   # LBA range (working set), 1 MiB .. the whole 1500 MiB file
    J = []
    for nm, rw in (("randread4k", "randread"), ("randwrite4k", "randwrite")):
        for sz in RANGES:
            J.append(Job("s6_range", f"{nm}_qd32_range{sz}", rw, "4k", 32, size=sz, runtime=8, ramp=2, trials=3))
    return J


def stage7():   # data pattern: incompressible vs compressible payload
    J = []
    for comp in (0, 90):
        J.append(Job("s7_pattern", f"randwrite4k_qd32_compress{comp}", "randwrite", "4k", 32, compress=comp, runtime=8, ramp=2, trials=3))
        J.append(Job("s7_pattern", f"seqwrite128k_qd32_compress{comp}", "write", "128k", 32, compress=comp, runtime=8, ramp=2, trials=3))
    return J


def stage8():   # misleading setups
    J = []
    for d in (1, 0):
        tag = "direct" if d else "buffered"
        J.append(Job("s8_pitfall", f"randread4k_qd1_{tag}", "randread", "4k", 1, direct=d, runtime=8, ramp=2, trials=3))
        J.append(Job("s8_pitfall", f"randread4k_qd32_{tag}", "randread", "4k", 32, direct=d, runtime=8, ramp=2, trials=3))
        J.append(Job("s8_pitfall", f"randwrite4k_qd1_{tag}", "randwrite", "4k", 1, direct=d, runtime=8, ramp=2, trials=3))
    # buffered reads depend on how much of the file the page cache already holds: with a 1 GiB range the first passes are mostly cold (the SSD is
    # read); a sequential buffered read of the range before the job (prewarm) puts it into the cache -> the cache-warm case (what a "small test file" benchmark measures)
    J.append(Job("s8_pitfall", "randread4k_qd1_buffered_warm", "randread", "4k", 1, direct=0, runtime=8, ramp=2, trials=3, prewarm=True))
    # effective queue depth. FIO on Windows REFUSES direct I/O with the synchronous engines, so the
    # synchronous-engine cases below are necessarily buffered: iodepth=32 has no effect on a synchronous
    # engine (effective QD = 1 per thread).
    J.append(Job("s8_pitfall", "randread4k_ioengine_sync_iodepth32_buffered", "randread", "4k", 32, ioengine="sync", direct=0, runtime=8, ramp=2, trials=3))
    J.append(Job("s8_pitfall", "randread4k_ioengine_sync_numjobs32_buffered", "randread", "4k", 1, numjobs=32, ioengine="sync", direct=0, runtime=8, ramp=2, trials=3))
    # direct I/O: outstanding I/Os = numjobs x iodepth, whichever way it is reached
    J.append(Job("s8_pitfall", "randread4k_windowsaio_iodepth32", "randread", "4k", 32, runtime=8, ramp=2, trials=3))
    J.append(Job("s8_pitfall", "randread4k_windowsaio_numjobs32_iodepth1", "randread", "4k", 1, numjobs=32, runtime=8, ramp=2, trials=3))
    return J


def rest(tag, seconds, dry):
    marker = os.path.join(RAW, f"rest_{tag}.done")
    if not dry and not os.path.exists(marker):
        print(f"  resting {seconds} s (idle, lets the controller fold/garbage-collect and the drive cool)", flush=True); time.sleep(seconds)
        open(marker, "w").write("1")


def stage9(dry=False):   # device state: burst vs steady state / SLC cache (heavy writes, runs near LAST)
    def quick(tag):
        return [Job("s9_state", f"{tag}_randwrite4k_qd1", "randwrite", "4k", 1, runtime=20, ramp=2, trials=3),
                Job("s9_state", f"{tag}_randread4k_qd1", "randread", "4k", 1, runtime=20, ramp=2, trials=3)]
    run_stage(quick("a_fresh"), dry)
    # ~450 GB written at ~505 MB/s over the whole 1500 MiB file; 1 s logs; SMART temperature joins by wall-clock
    run_stage([Job("s9_state", "b_sustained_seqwrite128k_qd32", "write", "128k", 32, size="1500m", runtime=900, ramp=0, trials=1, logs=True)], dry)
    run_stage(quick("c_after_seqwrite"), dry)
    run_stage([Job("s9_state", "d_sustained_randwrite4k_qd32", "randwrite", "4k", 32, size="1500m", runtime=600, ramp=0, trials=1, logs=True)], dry)
    run_stage(quick("e_after_randwrite"), dry)
    rest("s9", 300, dry)
    run_stage(quick("f_after_rest300s"), dry)


def stage10(dry=False):   # read-throughput stability: sustained QD32 random reads, two runs separated by an idle rest, OS counters on
    j = lambda name, rt: Job("s10_readstab", name, "randread", "4k", 32, runtime=rt, ramp=0, trials=1, logs=True)
    run_job(j("a_sustained_600s", 600), 0, dry, monitor=True)
    rest("s10", 300, dry)
    run_job(j("b_after_rest300s_600s", 600), 0, dry, monitor=True)


def stage11(dry=False):   # write cliff: stage 9 showed GC-limited slow episodes after ~180 GB of random writes -> characterise them
    j = lambda name, rw, bs, rt: Job("s11_cliff", name, rw, bs, 32, size="1500m", runtime=rt, ramp=0, trials=1, logs=True)
    run_job(j("a_randwrite4k_sustained_1200s", "randwrite", "4k", 1200), 0, dry)
    run_job(j("b_seqwrite128k_right_after_240s", "write", "128k", 240), 0, dry)
    rest("s11", 600, dry)
    run_job(j("c_randwrite4k_after_rest600s_600s", "randwrite", "4k", 600), 0, dry)


def stage12(dry=False):   # what triggers the read-throughput step? range / read count / time since the data was written?
    """Every run starts with a full rewrite of the file (fresh data), then an idle rest, then sustained QD-N random reads with 1 s logs.
    (a) ranges 16 MiB .. 1 GiB at QD32, 60 s rest: does the time to the first step scale with the range (per-block read counter)?
    (b) QD4 instead of QD32 (a third of the IOPS), on 1 GiB and on 16 MiB: does it scale with the number of reads (read counter) or not (time trigger)?
    (c) 600 s rest instead of 60 s: if the step is a background process that starts a fixed time after the write (e.g. SLC->TLC migration), the
        slow state should already be present when reading starts."""
    for sz, qd, rt, rest_s in (("16m", 32, 240, 60), ("64m", 32, 240, 60), ("256m", 32, 240, 60), ("1024m", 32, 240, 60), ("1024m", 4, 600, 60), ("16m", 32, 120, 600), ("16m", 4, 600, 60), ("16m", 32, 300, 600)):
        name = f"randread4k_qd{qd}_range{sz}_{rt}s" + ("" if rest_s == 60 else f"_rest{rest_s}s")
        job = Job("s12_readcount", name, "randread", "4k", qd, size=sz, runtime=rt, ramp=0, trials=1, logs=True)
        out = os.path.join(JSON_DIR, f"s12_readcount__{job.name}__t0.json")
        if os.path.exists(out) and os.path.getsize(out) > 0:
            continue
        if not dry:
            r = subprocess.run([FIO, os.path.join(JOBS, "precondition.fio"), "--output-format=json", f"--output={os.path.join(JSON_DIR, f's12_rewrite_{name}.json')}"], cwd=RAW, capture_output=True, text=True)
            assert r.returncode == 0, r.stdout + r.stderr
            print(f"  file rewritten twice before {name}; resting {rest_s} s", flush=True); time.sleep(rest_s)
        run_job(job, 0, dry)


STAGES = {1: stage1, 2: stage2, 3: stage3, 4: stage4, 5: stage5, 6: stage6, 7: stage7, 8: stage8}
SPECIAL = {9: stage9, 10: stage10, 11: stage11, 12: stage12}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2,3,4,5,6,7,8,9,10,12,11")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-smart-log", action="store_true")
    a = ap.parse_args()
    safety_check()
    precondition(a.dry_run)
    if not a.dry_run and not a.no_smart_log:
        SMART = SmartProbe(); SMART.query("start")
    for s in [int(x) for x in a.stages.split(",")]:
        print(f"== stage {s} ==", flush=True)
        if s in SPECIAL:
            SPECIAL[s](a.dry_run)
        else:
            run_stage(STAGES[s](), a.dry_run)
    print("all requested stages complete")
