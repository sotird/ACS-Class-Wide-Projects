#!/usr/bin/env python3
"""Run-to-run (process-to-process) reproducibility check.

The experiment binaries take repetitions inside one process (their IQR bands
measure within-run noise). This script compares the per-point MEDIANS across
three independent full runs (data/raw = run1, data/repeats/run2, run3) of the
whole suite, to bound the variation between separate launches (different
page placement, different scheduler decisions, different thermal moment).
"""
import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = {"run1": os.path.join(ROOT, "data", "raw"), "run2": os.path.join(ROOT, "data", "repeats", "run2"),
        "run3": os.path.join(ROOT, "data", "repeats", "run3")}
PROC = os.path.join(ROOT, "data", "processed"); GEN = os.path.join(ROOT, "report", "generated")
os.makedirs(GEN, exist_ok=True)

rows = []
def add(label, vals):
    v = np.array(vals, float)
    rows.append({"point": label, "run1": v[0], "run2": v[1], "run3": v[2], "mean": v.mean(), "cv_pct": 100 * v.std(ddof=1) / v.mean()})

def med(run, f, flt, col):
    d = pd.read_csv(os.path.join(RUNS[run], f)); d = d.query(flt); return float(d[col].median())

for b in ("scalar", "avx512", "avx512_fm"):
    for k in ("saxpy", "dot", "stencil"):
        if b == "avx512_fm" and k != "dot": continue
        for lvl, n in (("L1", 4096), ("L2", 65536), ("DRAM", 1 << 26)):
            add(f"exp1 {k} f32 {b} {lvl} GFLOP/s",
                [med(r, f"exp1_{b}.csv", f"kernel=='{k}' and dtype=='f32' and n=={n}", "gflops") for r in RUNS])
for n in (16, 17, 100):
    add(f"exp3 tail n={n} avx512 ns/call", [1e9 * med(r, "exp3_avx512.csv", f"sweep=='tail' and n=={n}", "seconds") for r in RUNS])
for off in (0, 4, 32):
    add(f"exp3 align_L1 off={off} avx512 GFLOP/s", [med(r, "exp3_avx512.csv", f"sweep=='align_L1' and byte_offset=={off}", "gflops") for r in RUNS])
for isa in ("scalar_fma", "avx2_fma", "avx512_fma"):
    add(f"exp7b peak {isa} GFLOP/s", [med(r, "exp7b_peak_flops.csv", f"isa=='{isa}'", "gflops") for r in RUNS])
for v in ("via_call", "inlined"):
    add(f"exp8 {v} n=4096 GFLOP/s", [med(r, "exp8_negative.csv", f"variant=='{v}' and n==4096", "gflops") for r in RUNS])
for pat in ("stride1", "stride8", "gather"):
    for reg in ("L2", "DRAM"):
        add(f"exp4 {pat} {reg} avx512 ns/out", [1e9 * med(r, "exp4_avx512.csv", f"regime=='{reg}' and pattern=='{pat}'", "seconds") / med(r, "exp4_avx512.csv", f"regime=='{reg}' and pattern=='{pat}'", "n_out") for r in RUNS])

df = pd.DataFrame(rows)
df.to_csv(os.path.join(PROC, "run_to_run_variation.csv"), index=False)
print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nmedian CV {df.cv_pct.median():.2f}%   mean CV {df.cv_pct.mean():.2f}%   max CV {df.cv_pct.max():.2f}% ({df.loc[df.cv_pct.idxmax(), 'point']})")
with open(os.path.join(GEN, "run_to_run.tex"), "w") as f:
    f.write(f"\\newcommand{{\\runcvmedian}}{{{df.cv_pct.median():.1f}}}\n\\newcommand{{\\runcvmean}}{{{df.cv_pct.mean():.1f}}}\n")
    f.write(f"\\newcommand{{\\runcvmax}}{{{df.cv_pct.max():.1f}}}\n\\newcommand{{\\runcvmaxpoint}}{{{df.loc[df.cv_pct.idxmax(), 'point'].replace('_', ' ').replace('%', '')}}}\n")
