#!/usr/bin/env python3
"""Correctness validation for Project 3.

1. Parser check: a synthetic FIO JSON with KNOWN values must be parsed back
   exactly (IOPS, MB/s, mean latency, percentiles, units).
2. Data identities on the REAL measurements:
   a. bandwidth == IOPS x block size (per run; catches unit mistakes),
   b. Little's Law: IOPS x mean latency ~= requested queue depth for every
      direct-I/O windowsaio run with iodepth >= 1 (catches latency-accounting
      errors and shows when the effective queue depth differs from requested),
   c. FIO's QD=1 latency vs. the independent C++ synchronous reference tool.
"""
import glob, importlib.util, json, os, sys, tempfile
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")

# ---- 1. synthetic JSON through the real parser ----------------------------------
src = open(os.path.join(ROOT, "scripts", "process_and_plot.py"), encoding="utf8").read()
start = src.index("PCT = {"); end = src.index("recs = []")        # PCT table + parse_json, exactly as used in the analysis
ns = {"np": np, "json": json}; exec(src[start:end], ns)                       # reuse the exact parser code
fake = {"global options": {"bs": "4k", "iodepth": "8", "rw": "randread", "direct": "1", "ioengine": "windowsaio"},
        "jobs": [{"job options": {"numjobs": "1"}, "usr_cpu": 1.0, "sys_cpu": 2.0,
                  "read": {"total_ios": 1000, "iops": 12345.0, "bw_bytes": 12345 * 4096,
                           "lat_ns": {"mean": 64000.0, "min": 50000, "max": 900000, "stddev": 5000.0,
                                      "percentile": {"50.000000": 63000, "99.000000": 120000, "99.900000": 400000}},
                           "clat_ns": {"mean": 63000.0}},
                  "write": {"total_ios": 0}}]}
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
    f.write("fio: some warning line\n" + json.dumps(fake))
rows = ns["parse_json"](f.name); os.unlink(f.name)
assert len(rows) == 1, "write direction with 0 I/Os must be dropped"
r = rows[0]
assert abs(r["iops"] - 12345) < 1e-9 and abs(r["mbps"] - 12345 * 4096 / 1e6) < 1e-9
assert abs(r["lat_mean_us"] - 64.0) < 1e-9 and abs(r["p50_us"] - 63.0) < 1e-9 and abs(r["p99_us"] - 120.0) < 1e-9 and abs(r["p99_9_us"] - 400.0) < 1e-9
assert r["iodepth"] == 8 and r["bs"] == "4k"
print("parser self-test on synthetic FIO JSON: PASS (units: ns->us, bytes->MB/s, percentiles, warning-prefixed JSON)")

# ---- 2. identities on the real data ---------------------------------------------
p = os.path.join(PROC, "all_runs_parsed.csv")
if not os.path.exists(p):
    print("no parsed data yet (run process_and_plot.py first) -- skipping data identities"); sys.exit(0)
df = pd.read_csv(p)
def kib(b): return {"k": 1024, "m": 1024 ** 2}[str(b)[-1]] * float(str(b)[:-1])
df["bs_bytes"] = df.bs.map(kib)
bw_err = (df.mbps * 1e6 / (df.iops * df.bs_bytes) - 1).abs()
print(f"(a) bandwidth = IOPS x block size: max deviation {100*bw_err.max():.2f}% over {len(df)} run-directions -> {'PASS' if bw_err.max() < 0.02 else 'FAIL'}")
d = df[(df.direct == 1) & (df.ioengine == "windowsaio") & (df.numjobs == 1) & (df.stage.isin(["s4_qd", "s5_tail", "s2_bs", "s1_zq"]))].copy()
d["N"] = d.iops * d.lat_mean_us * 1e-6
mixed = d.groupby(["stage", "name", "trial"]).N.transform("sum")      # rw mixes: N is the sum over read+write
d["N_total"] = mixed
ratio = (d.groupby(["stage", "name", "trial"]).N_total.first() / d.groupby(["stage", "name", "trial"]).iodepth.first())
print(f"(b) Little's Law  IOPS x mean latency / iodepth: median {ratio.median():.3f}, 5th-95th pct {ratio.quantile(.05):.3f}-{ratio.quantile(.95):.3f}"
      f" over {len(ratio)} runs -> {'PASS' if 0.85 < ratio.median() < 1.1 else 'CHECK'}")
if os.path.exists(os.path.join(PROC, "zero_queue_table.csv")):
    z = pd.read_csv(os.path.join(PROC, "zero_queue_table.csv"))
    if "ref_p50_us" in z:
        dev = ((z.p50_us / z.ref_p50_us) - 1).abs()
        print("(c) FIO vs C++ reference, QD=1 p50 latency deviation: " + ", ".join(f"{w} {100*v:.0f}%" for w, v in zip(z.workload, dev)))
