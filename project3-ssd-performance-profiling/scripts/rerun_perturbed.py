#!/usr/bin/env python3
"""Delete the outputs of every FIO job whose wall-clock interval overlaps a window listed in
data/raw/perturbation_windows.txt (times when other work on the host -- hardware-counter validation for Projects 1/2 --
may have disturbed the SSD benchmark) and re-run exactly those jobs (run_fio.py is resume-safe).
The deleted jobs are listed in data/raw/perturbed_jobs_rerun.txt so that the exclusion is documented.
Usage: python rerun_perturbed.py [--dry-run]"""
import csv
import glob
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
dry = "--dry-run" in sys.argv
wins = []
for line in open(os.path.join(RAW, "perturbation_windows.txt")):
    if line.strip() and not line.startswith("#"):
        a, b = line.split()[:2]; wins.append((float(a), float(b)))
hit = []
last = {}
for row in csv.DictReader(open(os.path.join(RAW, "job_times.csv"))):
    last.setdefault(row["job"], []).append((float(row["start_epoch_s"]), float(row["end_epoch_s"])))
for job, spans in last.items():
    s, e = spans[0]                       # the FIRST execution of the job is the one that overlapped (a re-run is recorded after it)
    if any(s <= w1 and e >= w0 for w0, w1 in wins) and len(spans) == 1:
        hit.append(job)
print(f"{len(hit)} jobs overlap a perturbation window")
stages = sorted({int(j.split("__")[0][1:].split("_")[0]) for j in hit})
if dry or not hit:
    print("stages to re-run:", stages); sys.exit(0)
with open(os.path.join(RAW, "perturbed_jobs_rerun.txt"), "a") as f:
    for job in hit:
        f.write(job + "\n")
        for p in (os.path.join(RAW, "json", job + ".json"),) + tuple(glob.glob(os.path.join(RAW, "logs", job + "_*"))):
            if os.path.exists(p): os.remove(p)
subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_fio.py"), "--stages", ",".join(map(str, stages))], check=True)
