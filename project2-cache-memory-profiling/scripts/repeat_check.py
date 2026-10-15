#!/usr/bin/env python3
"""Run-to-run (process-to-process) reproducibility of the headline Project 2
numbers: data/raw (run1) vs data/repeats/run2, run3 (headline experiments only:
exp1 on CPU 4, exp3, exp4, exp5, exp6). Within-run noise is the IQR band in
the figures; this bounds the variation between separate launches."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"run1": os.path.join(ROOT, "data", "raw"), "run2": os.path.join(ROOT, "data", "repeats", "run2"), "run3": os.path.join(ROOT, "data", "repeats", "run3")}
PROC = os.path.join(ROOT, "data", "processed"); GEN = os.path.join(ROOT, "report", "generated")
os.makedirs(GEN, exist_ok=True)
rows = []
def add(label, vals):
    v = np.array(vals, float); rows.append({"point": label, "run1": v[0], "run2": v[1], "run3": v[2], "cv_pct": 100 * v.std(ddof=1) / v.mean()})
def rd(run, f): return pd.read_csv(os.path.join(RUNS[run], f))

for layout in ("random_lines",):
    for lvl, (lo, hi) in {"L1": (4096, 32768), "L2": (131072, 786432), "L3": (2 << 20, 8 << 20), "DRAM": (512 << 20, 1 << 30)}.items():
        add(f"exp1 {layout} {lvl} ns", [rd(r, "exp1_latency_cpu4.csv").query(f"layout=='{layout}' and footprint_bytes>={lo} and footprint_bytes<={hi}").ns_per_hop.median() for r in RUNS])
for k in ("read_avx512", "write_avx512", "naive_fp_sum"):
    for lvl, (lo, hi) in {"L1": (4096, 40960), "L2": (102400, 716800), "DRAM": (1 << 30, 1 << 31)}.items():
        add(f"exp5 {k} {lvl} GB/s", [rd(r, "exp5_workingset.csv").query(f"kernel=='{k}' and footprint_bytes>={lo} and footprint_bytes<={hi}").gbps.median() for r in RUNS])
for mix in ("idle", "100R", "100W", "100W_NT", "70R30W", "50R50W"):
    for T in ((0,) if mix == "idle" else (1, 14)):
        d = [rd(r, "exp3_rw_mix.csv").query(f"mix=='{mix}' and threads=={T}") for r in RUNS]
        if mix != "idle": add(f"exp3 {mix} T={T} GB/s", [x.gbps.median() for x in d])
        add(f"exp3 {mix} T={T} probe p50 ns", [x.lat_p50_ns.median() for x in d])
for t in (0, 14):
    for mix in ("100R", "70R30W"):
        d = [rd(r, "exp4_intensity.csv").query(f"mix=='{mix}' and sweep=='delay' and delay_us=={0 if t else 800}") for r in RUNS]
        add(f"exp4 {mix} delay={'0' if t else '800us'} GB/s", [x.gbps.median() for x in d]); add(f"exp4 {mix} delay={'0' if t else '800us'} p50 ns", [x.lat_p50_ns.median() for x in d])
for p in ("seq_indep", "rand_indep", "rand_dep"):
    for lvl, fp in (("L2", 393216), ("DRAM", 1 << 30)):
        add(f"exp6 {p} ns/line @~{lvl}", [rd(r, "exp6_cachemiss.csv").query(f"pattern=='{p}'").assign(d=lambda x: (x.footprint_bytes - fp).abs()).sort_values("d").head(7).ns_per_line.median() for r in RUNS])
df = pd.DataFrame(rows); df.to_csv(os.path.join(PROC, "run_to_run_variation.csv"), index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nmedian CV {df.cv_pct.median():.2f}%  mean {df.cv_pct.mean():.2f}%  max {df.cv_pct.max():.2f}% ({df.loc[df.cv_pct.idxmax(), 'point']})")
with open(os.path.join(GEN, "run_to_run.tex"), "w") as f:
    f.write(f"\\newcommand{{\\runcvmedian}}{{{df.cv_pct.median():.1f}}}\n\\newcommand{{\\runcvmean}}{{{df.cv_pct.mean():.1f}}}\n\\newcommand{{\\runcvmax}}{{{df.cv_pct.max():.1f}}}\n")
    f.write(f"\\newcommand{{\\runcvmaxpoint}}{{{df.loc[df.cv_pct.idxmax(), 'point'].replace('_', ' ').replace('%', '').replace('~', '')}}}\n\\newcommand{{\\runcvn}}{{{len(df)}}}\n")
