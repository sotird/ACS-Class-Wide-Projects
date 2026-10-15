#!/usr/bin/env python3
"""Stage 10 (investigation of an anomaly): why does 4 KiB random-read QD32 throughput hop between plateaus?
Sustained QD32 random read with 1-second IOPS logs, separated by idle rests; `3` additionally rewrites the whole scratch file
first (does placement/age of the data matter?) and records OS counters (monitor_os.ps1) next to FIO.
Drive temperature / SMART are not readable without administrator rights, so this tests the symptom, not a cause.
Usage: run_read_stability_probe.py 1|2|3"""
import os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_fio as r

r.safety_check(); r.precondition()
J = lambda name, rt: r.Job("s10_thermal", name, "randread", "4k", 32, runtime=rt, ramp=0, trials=1, logs=True)
which = sys.argv[1] if len(sys.argv) > 1 else "1"
MON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "monitor_os.ps1")
if which == "1":
    r.run_job(J("a_sustained_240s", 240), 0)
    print("resting 180 s idle", flush=True); time.sleep(180)
    r.run_job(J("b_after_rest180s_90s", 90), 0)
elif which == "2":
    r.run_job(J("c_sustained_420s_after_prior", 420), 0)
    print("resting 300 s idle", flush=True); time.sleep(300)
    r.run_job(J("d_after_rest300s_300s", 300), 0)    # (OS counters for this run were recorded with an earlier monitor_os.ps1 version)
else:
    # rewrite the whole file twice (same job as the preconditioning), then read for 360 s while recording OS counters
    pre = subprocess.run([r.FIO, os.path.join(r.JOBS, "precondition.fio"), "--output-format=json",
                          f"--output={os.path.join(r.JSON_DIR, 's10_rewrite_precondition.json')}"], cwd=r.RAW, capture_output=True, text=True)
    assert pre.returncode == 0, pre.stdout + pre.stderr
    print("file rewritten twice", flush=True)
    mon = subprocess.Popen(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", MON, "-Seconds", "366", "-Out",
                            os.path.join(r.RAW, "logs", "s10_os_monitor_e.csv")], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    r.run_job(J("e_after_rewrite_360s", 360), 0)
    mon.wait()
print("done")
