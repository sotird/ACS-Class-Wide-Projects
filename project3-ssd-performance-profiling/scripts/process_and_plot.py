#!/usr/bin/env python3
"""Parse raw FIO JSON (+ C++ reference tool CSVs + time-series logs) into
processed tables, 5 figures and LaTeX tables.

Statistics: each (configuration) is run for several trials; plots show the
MEDIAN over trials with min--max whiskers/bands. Latencies are FIO's total
submission-to-completion latency (lat_ns) percentiles unless noted.
"""
import glob
import json
import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, PROC, RES = (os.path.join(ROOT, d) for d in ("data/raw", "data/processed", "results"))
RANGES = ("1m", "4m", "16m", "64m", "256m", "1024m", "1500m")
GEN = os.path.join(ROOT, "report", "generated")
for d in (PROC, RES, GEN):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
C = {"blue": "#4c72b0", "orange": "#dd8452", "green": "#55a868", "red": "#c44e52", "purple": "#8172b3", "gray": "#777777", "teal": "#17becf"}
PCT = {"50.000000": "p50", "75.000000": "p75", "90.000000": "p90", "95.000000": "p95", "99.000000": "p99",
       "99.500000": "p99_5", "99.900000": "p99_9", "99.990000": "p99_99", "99.999000": "p99_999"}


def tex(s): return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


# ------------------------------------------------------------------ parse ----
def parse_json(path):
    txt = open(path, encoding="utf8", errors="replace").read()
    j = json.loads(txt[txt.index("{"):])
    job = j["jobs"][0]
    opts = {**j.get("global options", {}), **job.get("job options", {})}
    rows = []
    for d in ("read", "write"):
        x = job[d]
        if x["total_ios"] == 0:
            continue
        lat = x["lat_ns"]; clat = x["clat_ns"]
        r = {"dir": d, "iops": x["iops"], "mbps": x["bw_bytes"] / 1e6, "ios": x["total_ios"],
             "lat_mean_us": lat["mean"] / 1e3, "lat_min_us": lat["min"] / 1e3, "lat_max_us": lat["max"] / 1e3, "lat_std_us": lat["stddev"] / 1e3,
             "clat_mean_us": clat["mean"] / 1e3, "usr_cpu": job.get("usr_cpu", np.nan), "sys_cpu": job.get("sys_cpu", np.nan)}
        for k, v in lat.get("percentile", {}).items():
            r[PCT.get(k, k) + "_us"] = v / 1e3
        r.update({"bs": opts.get("bs"), "rw": opts.get("rw"), "iodepth": int(opts.get("iodepth", 1)), "numjobs": int(opts.get("numjobs", 1)),
                  "mix": opts.get("rwmixread"), "direct": int(opts.get("direct", 1)), "ioengine": opts.get("ioengine"), "size": opts.get("size")})
        rows.append(r)
    return rows


recs = []
for p in sorted(glob.glob(os.path.join(RAW, "json", "*.json"))):
    base = os.path.basename(p)[:-5]
    if base.count("__") != 2 or os.path.getsize(p) == 0:      # preconditioning jobs are not measurements; empty = still running / aborted
        continue
    stage, name, t = base.split("__")
    for r in parse_json(p):
        recs.append({"stage": stage, "name": name, "trial": int(t[1:]), **r})
df = pd.DataFrame(recs)
df.to_csv(os.path.join(PROC, "all_runs_parsed.csv"), index=False)
print(f"parsed {df.groupby(['stage','name','trial']).ngroups} runs from {len(glob.glob(os.path.join(RAW, 'json', '*.json')))} JSON files")

# stall census: runs whose maximum latency exceeds 100 ms and 50x their median (device-level stalls, not queueing)
st = df[(df.lat_max_us > 100000) & (df.lat_max_us > 50 * df.p50_us)].copy()
st["job"] = st.stage + " / " + st.name + " / t" + st.trial.astype(str)
st[["job", "dir", "iops", "p50_us", "p99_us", "lat_max_us"]].sort_values("lat_max_us", ascending=False).to_csv(os.path.join(PROC, "stall_census.csv"), index=False)
NL, BSL = chr(10), chr(92)
with open(os.path.join(GEN, "table_stalls.tex"), "w") as f:
    f.write(BSL + "begin{tabular}{lrrrr}" + NL + BSL + "toprule" + NL + "job (stage / configuration / trial) & IOPS & p50 $" + BSL + "mu$s & p99 $" + BSL + "mu$s & max ms " + BSL + BSL + NL + BSL + "midrule" + NL)
    for r in st.sort_values("lat_max_us", ascending=False).head(14).itertuples():
        f.write(f"{tex(r.job)} & {r.iops:,.0f} & {r.p50_us:.0f} & {r.p99_us:,.0f} & {r.lat_max_us / 1000:.0f} " + BSL + BSL + NL)
    f.write(BSL + "bottomrule" + NL + BSL + "end{tabular}" + NL)
print(f"stall census: {len(st)} of {len(df)} run-directions have a stall (max > 100 ms and > 50x p50)")
PC = [c for c in df.columns if re.match(r"p\d", c)] + ["lat_mean_us", "iops", "mbps", "lat_max_us"]
def agg(d, extra_keys=()):
    keys = ["stage", "name", "dir", *extra_keys]
    g = d.groupby(keys, dropna=False)       # rwmixread is absent (NaN) for pure read/write jobs: keep those groups
    out = g[PC].median()
    for c in ("iops", "mbps", "p50_us", "p99_us", "lat_mean_us"):
        out[c + "_min"] = g[c].min(); out[c + "_max"] = g[c].max()
    out["trials"] = g.size()
    return out.reset_index()
A = agg(df, ("bs", "iodepth", "numjobs", "rw", "mix", "direct", "ioengine", "size"))
A.to_csv(os.path.join(PROC, "summary_by_config.csv"), index=False)

def get(stage, name, d="read"):
    r = A[(A.stage == stage) & (A.name == name) & (A.dir == d)]
    return r.iloc[0] if len(r) else None

# C++ reference tool results
ref = pd.concat([pd.read_csv(f) for f in glob.glob(os.path.join(RAW, "ref", "*.csv"))], ignore_index=True) if glob.glob(os.path.join(RAW, "ref", "*.csv")) else pd.DataFrame()
if len(ref):
    ref = ref[ref["mode"] != "mode"].copy()          # the C++ tool re-emits its header line on every append
    for c_ in ref.columns[1:]: ref[c_] = ref[c_].astype(float)
    refm = ref.groupby("mode").median(numeric_only=True).reset_index(); refm.to_csv(os.path.join(PROC, "ref_tool_summary.csv"), index=False)

# ------------------------------------------------------------ zero-queue ----
ZQ = [("randread4k_qd1", "read", "4 KiB random read"), ("randwrite4k_qd1", "write", "4 KiB random write"),
      ("seqread128k_qd1", "read", "128 KiB seq read"), ("seqwrite128k_qd1", "write", "128 KiB seq write")]
REFMAP = {"randread4k_qd1": "randread4k_direct", "randwrite4k_qd1": "randwrite4k_direct", "seqread128k_qd1": "seqread128k_direct", "seqwrite128k_qd1": "seqwrite128k_direct"}
zq_rows = []
for nm, d, lab in ZQ:
    r = get("s1_zq", nm, d)
    row = {"workload": lab, "iops": r.iops, "mbps": r.mbps, "mean_us": r.lat_mean_us, "p50_us": r.p50_us, "p95_us": r.p95_us, "p99_us": r.p99_us,
           "p99_9_us": r.p99_9_us, "p50_min": r.p50_us_min, "p50_max": r.p50_us_max, "p99_min": r.p99_us_min, "p99_max": r.p99_us_max}
    if len(ref):
        q = refm[refm["mode"] == REFMAP[nm]]
        if len(q):
            row.update({"ref_p50_us": float(q.p50_us.iloc[0]), "ref_p99_us": float(q.p99_us.iloc[0]), "ref_mean_us": float(q.mean_us.iloc[0])})
    zq_rows.append(row)
zq = pd.DataFrame(zq_rows); zq.to_csv(os.path.join(PROC, "zero_queue_table.csv"), index=False)
with open(os.path.join(GEN, "table_zeroqueue.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\nQD=1 workload & IOPS & MB/s & mean & p50 & p95 & p99 & p99.9 & C++ ref p50 / p99 \\\\\n & & & \\multicolumn{4}{c}{FIO latency ($\\mu$s)} & & ($\\mu$s) \\\\\n\\midrule\n")
    for r in zq.itertuples():
        rf = f"{r.ref_p50_us:.0f} / {r.ref_p99_us:.0f}" if hasattr(r, "ref_p50_us") and not pd.isna(getattr(r, "ref_p50_us", np.nan)) else "--"
        f.write(f"{r.workload} & {r.iops:,.0f} & {r.mbps:.1f} & {r.mean_us:.0f} & {r.p50_us:.0f} & {r.p95_us:.0f} & {r.p99_us:.0f} & {r.p99_9_us:.0f} & {rf} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ----------------------------------------------------------- block size -----
BSV = {"4k": 4, "16k": 16, "32k": 32, "64k": 64, "128k": 128, "256k": 256, "512k": 512, "1m": 1024}
SER = [("randread", "read", C["blue"], "-", "random read"), ("randwrite", "write", C["orange"], "-", "random write"),
       ("seqread", "read", C["green"], "--", "sequential read"), ("seqwrite", "write", C["red"], "--", "sequential write")]
bs_rows = []
for pat, dirn, col, ls, lab in SER:
    d = "read" if "read" in pat else "write"
    for bs, kib in BSV.items():
        for qd in (1, 32):
            r = get("s2_bs", f"{pat}_bs{bs}_qd{qd}", d)
            if r is None: continue
            bs_rows.append({"series": lab, "bs": bs, "kib": kib, "qd": qd, "iops": r.iops, "mbps": r.mbps, "p50_us": r.p50_us, "p99_us": r.p99_us, "mean_us": r.lat_mean_us})
bst = pd.DataFrame(bs_rows); bst.to_csv(os.path.join(PROC, "blocksize_matrix.csv"), index=False)
with open(os.path.join(GEN, "table_blocksize.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{QD=1} & \\multicolumn{3}{c}{QD=32} \\\\\npattern & bs & IOPS & MB/s & p50 $\\mu$s & IOPS & MB/s & mean $\\mu$s \\\\\n\\midrule\n")
    for _, _, _, _, lab in SER:
        for bs in BSV:
            a = bst[(bst.series == lab) & (bst.bs == bs) & (bst.qd == 1)]; b = bst[(bst.series == lab) & (bst.bs == bs) & (bst.qd == 32)]
            if len(a) and len(b):
                a, b = a.iloc[0], b.iloc[0]
                f.write(f"{lab if bs == '4k' else ''} & {bs} & {a.iops:,.0f} & {a.mbps:.0f} & {a.p50_us:.0f} & {b.iops:,.0f} & {b.mbps:.0f} & {b.mean_us:.0f} \\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
labs = [z[2] for z in ZQ]; x = np.arange(len(ZQ)); w = 0.2
for j, (pc, col) in enumerate((("p50_us", C["blue"]), ("p95_us", C["green"]), ("p99_us", C["orange"]), ("p99_9_us", C["red"]))):
    vals = [zq[pc].iloc[i] for i in range(len(zq))]
    ax.bar(x + (j - 1.5) * w, vals, w, color=col, label=pc.replace("_us", "").replace("p99_9", "p99.9").replace("p", "p"))
for i in range(len(zq)):
    if "ref_p50_us" in zq and not pd.isna(zq.ref_p50_us.iloc[i]):
        ax.plot(i - 1.5 * w, zq.ref_p50_us.iloc[i], "k^", ms=6); ax.plot(i - 0.5 * w + w, zq.ref_p99_us.iloc[i], "kv", ms=6)
ax.plot([], [], "k^", label="C++ reference p50"); ax.plot([], [], "kv", label="C++ reference p99")
ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n", 1) for l in labs], fontsize=8); ax.set_yscale("log"); ax.set_ylabel("latency (us)")
ax.set_title("Zero-queue (QD=1) latency percentiles, 5 trials", fontsize=9.5); ax.legend(fontsize=7, ncol=2)
for ax, ycol, yl, ttl in ((axes[0, 1], "mbps", "throughput (MB/s)", "Block-size sweep, QD=32: throughput"), (axes[1, 0], "iops", "IOPS", "Block-size sweep, QD=32: IOPS")):
    for _, _, col, ls, lab in SER:
        s = bst[(bst.series == lab) & (bst.qd == 32)].sort_values("kib")
        ax.plot(s.kib, s[ycol], ls, marker="o", ms=3.5, color=col, label=lab)
    ax.set_xscale("log", base=2); ax.set_yscale("log" if ycol == "iops" else "linear"); ax.set_xlabel("block size (KiB)"); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=9.5)
    if ycol == "mbps":
        ax.axhline(600, color="k", ls=":", lw=1); ax.text(4, 612, "SATA III 6 Gb/s raw payload ceiling (600 MB/s)", fontsize=7)
        ax.legend(fontsize=7, loc="center right")
ax = axes[1, 1]
for _, _, col, ls, lab in SER:
    s = bst[(bst.series == lab) & (bst.qd == 1)].sort_values("kib")
    ax.plot(s.kib, s.p50_us, ls, marker="o", ms=3.5, color=col, label=f"{lab} p50")
    ax.plot(s.kib, s.p99_us, ":", marker="x", ms=3.5, color=col, alpha=0.7)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("block size (KiB)"); ax.set_ylabel("QD=1 latency (us); dotted = p99"); ax.set_title("Block-size sweep, QD=1: latency", fontsize=9.5); ax.legend(fontsize=7)
fig.suptitle("Project 3 -- Fig. A: zero-queue baselines and block-size sweep (Crucial MX500 SATA, direct I/O, 1 GiB range)", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figA_baselines_and_blocksize.png"), bbox_inches="tight"); plt.close(fig)

# ----------------------------------------------------------- QD sweep -------
QDS = [1, 2, 4, 8, 16, 32, 64, 128, 256]
WL = [("randread4k", "read", C["blue"], "random 4K read"), ("randwrite4k", "write", C["orange"], "random 4K write")]
qd_rows = []
for nm, d, col, lab in WL:
    for qd in QDS:
        r = get("s4_qd", f"{nm}_qd{qd}", d)
        if r is not None: qd_rows.append({"wl": lab, "qd": qd, "iops": r.iops, "mbps": r.mbps, "mean_us": r.lat_mean_us, "p50_us": r.p50_us, "p99_us": r.p99_us,
                                          "iops_min": r.iops_min, "iops_max": r.iops_max})
for qd in QDS:   # 70/30 mix: combine read+write
    r, w_ = get("s4_qd", f"randmix70_4k_qd{qd}", "read"), get("s4_qd", f"randmix70_4k_qd{qd}", "write")
    if r is not None and w_ is not None:
        iops = r.iops + w_.iops
        mean = (r.iops * r.lat_mean_us + w_.iops * w_.lat_mean_us) / iops
        qd_rows.append({"wl": "random 4K 70/30 mix", "qd": qd, "iops": iops, "mbps": r.mbps + w_.mbps, "mean_us": mean, "p50_us": r.p50_us, "p99_us": max(r.p99_us, w_.p99_us),
                        "iops_min": r.iops_min + w_.iops_min, "iops_max": r.iops_max + w_.iops_max})
for nm, d, lab in (("seqread128k", "read", "seq 128K read"), ("seqwrite128k", "write", "seq 128K write")):
    for qd in (1, 2, 4, 8, 16, 32, 64):
        r = get("s4_qd", f"{nm}_qd{qd}", d)
        if r is not None: qd_rows.append({"wl": lab, "qd": qd, "iops": r.iops, "mbps": r.mbps, "mean_us": r.lat_mean_us, "p50_us": r.p50_us, "p99_us": r.p99_us, "iops_min": r.iops_min, "iops_max": r.iops_max})
qdt = pd.DataFrame(qd_rows); qdt["little_N"] = qdt.iops * qdt.mean_us * 1e-6
qdt.to_csv(os.path.join(PROC, "qd_sweep.csv"), index=False)
knee_rows = []
for wl, s in qdt.groupby("wl", sort=False):
    s = s.sort_values("qd"); mx = s.iops.max()
    k90 = s[s.iops >= 0.9 * mx].iloc[0]; k98 = s[s.iops >= 0.98 * mx].iloc[0]; one = s.iloc[0]
    base = {"random 4K read": ("randread4k_qd1", "read"), "random 4K write": ("randwrite4k_qd1", "write"), "seq 128K read": ("seqread128k_qd1", "read"),
            "seq 128K write": ("seqwrite128k_qd1", "write")}.get(wl)
    b0 = get("s1_zq", base[0], base[1]) if base else None          # QD1 reference = the 5-trial fresh-state baseline (stage 1), not a sweep point that a stall may have hit
    q1_iops, q1_mean = (b0.iops, b0.lat_mean_us) if b0 is not None else (one.iops, one.mean_us)
    knee_rows.append({"workload": wl, "qd1_iops": q1_iops, "qd1_mean_us": q1_mean, "max_iops": mx, "max_mbps": s.mbps.max(), "knee_qd_90": int(k90.qd), "knee_mean_us": k90.mean_us,
                      "sat_qd_98": int(k98.qd), "qd256_mean_us": s.mean_us.iloc[-1], "latency_x_at_top": s.mean_us.iloc[-1] / q1_mean, "top_qd": int(s.qd.iloc[-1])})
kn = pd.DataFrame(knee_rows); kn.to_csv(os.path.join(PROC, "knee_table.csv"), index=False)
with open(os.path.join(GEN, "table_knee.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\nworkload & QD1 IOPS & max IOPS & MB/s at max & $N^{*}=L_1 X_{\\max}$ & QD for 90\\% & mean latency there & QD for 98\\% & latency $\\times$ at top QD \\\\\n\\midrule\n")
    for r in kn.itertuples():
        f.write(f"{tex(r.workload)} & {r.qd1_iops:,.0f} & {r.max_iops:,.0f} & {r.max_mbps:.0f} & {r.qd1_mean_us * 1e-6 * r.max_iops:.1f} & {r.knee_qd_90} & {r.knee_mean_us:.0f} $\\mu$s & {r.sat_qd_98} & {r.latency_x_at_top:.0f}$\\times$ (QD {r.top_qd}) \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# numjobs sweep (total outstanding = numjobs x iodepth)
nj = []
for n_ in (1, 2, 4, 8, 16, 32):
    r = get("s4_nj", f"randread4k_qd1_nj{n_}", "read")
    if r is not None: nj.append({"outstanding": n_, "iops": r.iops, "mean_us": r.lat_mean_us})
njt = pd.DataFrame(nj)

# read/write mix
mix_rows = []
for label in ("100R", "100W", "70R30W", "50R50W"):
    for qd in (1, 32):
        rr, ww = get("s3_mix", f"rand4k_{label}_qd{qd}", "read"), get("s3_mix", f"rand4k_{label}_qd{qd}", "write")
        row = {"mix": label, "qd": qd, "iops": (rr.iops if rr is not None else 0) + (ww.iops if ww is not None else 0),
               "mbps": (rr.mbps if rr is not None else 0) + (ww.mbps if ww is not None else 0)}
        if rr is not None: row.update({"r_p50": rr.p50_us, "r_p99": rr.p99_us, "r_mean": rr.lat_mean_us})
        if ww is not None: row.update({"w_p50": ww.p50_us, "w_p99": ww.p99_us, "w_mean": ww.lat_mean_us})
        mix_rows.append(row)
mixt = pd.DataFrame(mix_rows); mixt.to_csv(os.path.join(PROC, "rw_mix_random4k.csv"), index=False)
with open(os.path.join(GEN, "table_mix.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\nmix (4 KiB random) & QD & IOPS & MB/s & read p50 / p99 $\\mu$s & write p50 / p99 $\\mu$s \\\\\n\\midrule\n")
    for r in mixt.itertuples():
        rd = f"{r.r_p50:.0f} / {r.r_p99:.0f}" if not pd.isna(getattr(r, "r_p50", np.nan)) else "--"
        wr = f"{r.w_p50:.0f} / {r.w_p99:.0f}" if not pd.isna(getattr(r, "w_p50", np.nan)) else "--"
        f.write(f"{tex(r.mix)} & {r.qd} & {r.iops:,.0f} & {r.mbps:.0f} & {rd} & {wr} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for wl, col in (("random 4K read", C["blue"]), ("random 4K write", C["orange"]), ("random 4K 70/30 mix", C["green"])):
    s = qdt[qdt.wl == wl].sort_values("qd")
    ax.plot(s.iops, s.mean_us, "o-", ms=4, color=col, label=f"{wl} (mean)")
    ax.plot(s.iops, s.p99_us, "x:", ms=4, color=col, alpha=0.8)
    for _, r in s.iterrows():
        if r.qd in (1, 4, 16, 32, 256): ax.annotate(f"QD{int(r.qd)}", (r.iops, r.mean_us), xytext=(4, 4), textcoords="offset points", fontsize=6.5, color=col)
    k = kn[kn.workload == wl].iloc[0]; ks = s[s.qd == k.knee_qd_90].iloc[0]
    ax.plot(ks.iops, ks.mean_us, "*", ms=14, color="k", zorder=5)
ax.set_yscale("log"); ax.set_xlabel("throughput (IOPS)"); ax.set_ylabel("latency (us): solid = mean, dotted = p99")
ax.set_title("Throughput-latency curve (QD swept 1..256); star = knee (first QD with >= 90% of max IOPS)", fontsize=9); ax.legend(fontsize=7)
ax = axes[0, 1]
for wl, col in (("random 4K read", C["blue"]), ("random 4K write", C["orange"]), ("random 4K 70/30 mix", C["green"])):
    s = qdt[qdt.wl == wl].sort_values("qd"); ax.errorbar(s.qd, s.iops, yerr=[s.iops - s.iops_min, s.iops_max - s.iops], fmt="o-", ms=3.5, capsize=2, color=col, label=wl)
ax.plot(njt.outstanding, njt.iops, "s--", ms=4, color=C["gray"], label="random 4K read via numjobs (QD1 each)")
ax.axvline(32, color="k", ls="--", lw=1); ax.text(33, ax.get_ylim()[0] * 1.5 if ax.get_ylim()[0] > 0 else 1, "SATA NCQ depth = 32", fontsize=7.5, rotation=90, va="bottom")
ax.set_xscale("log", base=2); ax.set_xlabel("outstanding I/Os (iodepth, or numjobs x iodepth)"); ax.set_ylabel("IOPS"); ax.set_title("IOPS vs queue depth / parallelism", fontsize=9.5); ax.legend(fontsize=7)
ax = axes[1, 0]
for wl, col in (("random 4K read", C["blue"]), ("random 4K write", C["orange"]), ("random 4K 70/30 mix", C["green"]), ("seq 128K read", C["red"]), ("seq 128K write", C["purple"])):
    s = qdt[qdt.wl == wl].sort_values("qd"); ax.plot(s.qd, s.little_N, "o-", ms=3.5, color=col, label=wl)
ax.plot([1, 256], [1, 256], "k--", lw=1, label="N = QD (identity)")
ax.set_xscale("log", base=2); ax.set_yscale("log", base=2); ax.set_xlabel("requested iodepth"); ax.set_ylabel("N = IOPS x mean latency (Little's Law)")
ax.set_title("Little's Law check: measured N vs requested queue depth", fontsize=9.5); ax.legend(fontsize=7)
ax = axes[1, 1]
mixes = ["100R", "100W", "70R30W", "50R50W"]; xx = np.arange(4)
for j, (qd, col) in enumerate(((1, C["blue"]), (32, C["orange"]))):
    v = [mixt[(mixt.mix == m) & (mixt.qd == qd)].iops.iloc[0] for m in mixes]
    ax.bar(xx + (j - 0.5) * 0.38, v, 0.38, color=col, label=f"QD{qd} IOPS")
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(mixes); ax.set_ylabel("IOPS (log)")
ax2 = ax.twinx(); ax2.grid(False)
for j, (qd, col) in enumerate(((1, "#1f3a93"), (32, "#a0471d"))):
    rp = [mixt[(mixt.mix == m) & (mixt.qd == qd)].get("r_p99", pd.Series([np.nan])).iloc[0] for m in mixes]
    wp = [mixt[(mixt.mix == m) & (mixt.qd == qd)].get("w_p99", pd.Series([np.nan])).iloc[0] for m in mixes]
    ax2.plot(xx + (j - 0.5) * 0.38, rp, "v", color=col, ms=7, label=f"read p99, QD{qd}"); ax2.plot(xx + (j - 0.5) * 0.38, wp, "^", color=col, ms=7, mfc="none", label=f"write p99, QD{qd}")
ax2.set_yscale("log"); ax2.set_ylabel("p99 latency (us)"); ax.set_title("Read/write mix (4 KiB random): IOPS and p99 latency", fontsize=9.5)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.5, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.1))
fig.suptitle("Project 3 -- Fig. B: queue-depth sweep (throughput-latency knee), Little's Law, read/write mix", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figB_qd_sweep_and_mix.png"), bbox_inches="tight"); plt.close(fig)

# ------------------------------------------------------------- tail ---------
tail_rows = []
for nm, d, lab in (("randread4k", "read", "random 4K read"), ("randwrite4k", "write", "random 4K write")):
    for qd in (1, 4, 16, 32, 128):
        r = get("s5_tail", f"{nm}_qd{qd}", d)
        if r is None: continue
        tail_rows.append({"wl": lab, "qd": qd, "iops": r.iops, "mean": r.lat_mean_us, **{p: r[p + "_us"] for p in ("p50", "p75", "p90", "p95", "p99", "p99_5", "p99_9", "p99_99", "p99_999")}, "max": r.lat_max_us, "iops_min": r.iops_min, "iops_max": r.iops_max,
                          "ios": float(df[(df.stage == "s5_tail") & (df.name == f"{nm}_qd{qd}") & (df.dir == d)].ios.median())})
tt = pd.DataFrame(tail_rows); tt.to_csv(os.path.join(PROC, "tail_latency.csv"), index=False)
with open(os.path.join(GEN, "table_tail.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrrr}\n\\toprule\nworkload & QD & IOPS, median (min--max of 2 trials) & p50 & p95 & p99 & p99.9 & p99.99 & max & p99/p50 \\\\\n & & & \\multicolumn{5}{c}{latency ($\\mu$s)} & & \\\\\n\\midrule\n")
    for r in tt.itertuples():
        f.write(f"{r.wl if r.qd == 1 else ''} & {r.qd} & {r.iops / 1000:,.1f}\\,k ({r.iops_min / 1000:,.1f}--{r.iops_max / 1000:,.1f}) & {r.p50:.0f} & {r.p95:.0f} & {r.p99:.0f} & {r.p99_9:.0f} & {r.p99_99:.0f} & {r.max:.0f} & {r.p99 / r.p50:.1f} \\\\\n")
        if r.qd == 128: f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
PCTX = [("p50", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("p99", 99), ("p99_5", 99.5), ("p99_9", 99.9), ("p99_99", 99.99), ("p99_999", 99.999)]
cm = plt.get_cmap("viridis")
for ax, lab in ((axes[0, 0], "random 4K read"), (axes[0, 1], "random 4K write")):
    for i, qd in enumerate((1, 4, 16, 32, 128)):
        r = tt[(tt.wl == lab) & (tt.qd == qd)]
        if not len(r): continue
        r = r.iloc[0]
        ax.plot([-np.log10(1 - p / 100) for _, p in PCTX], [r[k] for k, _ in PCTX], "o-", ms=3.5, color=cm(i / 4.5), label=f"QD{qd}")
    ax.set_xticks([np.log10(1 / (1 - p / 100)) for p in (50, 90, 99, 99.9, 99.99, 99.999)]); ax.set_xticklabels(["p50", "p90", "p99", "p99.9", "p99.99", "p99.999"])
    ax.set_yscale("log"); ax.set_ylabel("latency (us)"); ax.set_title(f"Latency percentile profile: {lab}", fontsize=9.5); ax.legend(fontsize=7)
ax = axes[1, 0]
for lab, col in (("random 4K read", C["blue"]), ("random 4K write", C["orange"])):
    s = tt[tt.wl == lab].sort_values("qd")
    ax.plot(s.qd, s.p99 / s.p50, "o-", color=col, label=f"{lab}: p99 / p50"); ax.plot(s.qd, s.p99_9 / s.p50, "s--", color=col, alpha=0.7, label="p99.9 / p50")
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("queue depth"); ax.set_ylabel("tail / median latency ratio"); ax.set_title("Tail amplification vs queue depth", fontsize=9.5); ax.legend(fontsize=7)
ax = axes[1, 1]
for lab, col in (("random 4K read", C["blue"]), ("random 4K write", C["orange"])):
    s = tt[tt.wl == lab].sort_values("qd")
    ax.plot(s.iops, s.p99_9, "o-", color=col, label=f"{lab}: p99.9"); ax.plot(s.iops, s.p50, "s:", color=col, alpha=0.7, label=f"{lab}: p50")
    for _, r in s.iterrows(): ax.annotate(f"QD{int(r.qd)}", (r.iops, r.p99_9), xytext=(4, 3), textcoords="offset points", fontsize=7)
ax.set_yscale("log"); ax.set_xlabel("achieved IOPS"); ax.set_ylabel("latency (us)"); ax.set_title("Tail latency vs delivered throughput (QoS curve)", fontsize=9.5); ax.legend(fontsize=7)
fig.suptitle("Project 3 -- Fig. C: tail latency (p50..p99.999) at mid-QD, near-knee QD and beyond", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figC_tail_latency.png"), bbox_inches="tight"); plt.close(fig)

# ------------------------------------- device state / read stability / pitfalls ---
def series(prefix, kind):
    fs = sorted(glob.glob(os.path.join(RAW, "logs", f"{prefix}__t0_{kind}.*.log")))
    if not fs: return None
    return pd.read_csv(fs[0], header=None, names=["msec", "val", "dir", "bs", "off"], skipinitialspace=True)


SMART_P = os.path.join(RAW, "logs", "smart_log.csv"); JT_P = os.path.join(RAW, "job_times.csv")
SMART = pd.read_csv(SMART_P) if os.path.exists(SMART_P) else None
if SMART is not None: SMART.loc[SMART.temperature_c <= 0, "temperature_c"] = np.nan      # a failed query returns 0
JT = pd.read_csv(JT_P).drop_duplicates("job", keep="last").set_index("job") if os.path.exists(JT_P) else None


def temp_series(base):
    if SMART is None or JT is None or base not in JT.index: return None
    s0, s1 = JT.loc[base, "start_epoch_s"], JT.loc[base, "end_epoch_s"]
    m = SMART[(SMART.epoch_s >= s0 - 3) & (SMART.epoch_s <= s1 + 1)].dropna(subset=["temperature_c"])
    return pd.DataFrame({"t": m.epoch_s - s0, "temp": m.temperature_c, "tag": m.tag})


sust_rows = []


def ts_panel(ax, base, kind, scale, color, ylabel, title, label, logy=False):
    s = series(base, kind)
    if s is None:
        ax.text(0.5, 0.5, f"(no data: {label})", ha="center", transform=ax.transAxes); ax.set_title(title, fontsize=9.5); return None
    t = s.msec.values / 1000; v = s.val.values.astype(float) * scale
    ax.plot(t, v, color=color, lw=0.6, alpha=0.45); ax.plot(t, pd.Series(v).rolling(10, center=True, min_periods=1).median(), color=color, lw=1.6, label=f"{ylabel.split(' (')[0]} (1 s samples; 10 s median)")
    ax.set_xlabel("time since start of run (s)"); ax.set_ylabel(ylabel); ax.set_title(title, fontsize=9.5)
    if logy: ax.set_yscale("log"); ax.set_ylim(100, v.max() * 2)
    else: ax.set_ylim(0, v.max() * 1.12)
    tt = temp_series(base + "__t0")
    row = {"series": label, "seconds": len(v), "first30s": float(np.median(v[:30])), "last30s": float(np.median(v[-30:])), "min": float(v.min()), "max": float(v.max()),
           "pct_seconds_below_75pct_of_p95": float(100 * np.mean(v < 0.75 * np.percentile(v, 95)))}
    if tt is not None and len(tt):
        ax2 = ax.twinx(); ax2.plot(tt.t, tt.temp, "k.-", ms=3, lw=0.9, label="SSD temperature (C), right"); ax2.set_ylabel("SSD temperature (C)"); ax2.grid(False)
        ax2.set_ylim(min(30, tt.temp.min() - 2), max(tt.temp.max() + 3, 50))
        ax.plot(tt.t[tt.tag.str.startswith("during")], np.full((tt.tag.str.startswith("during")).sum(), ax.get_ylim()[0]), "|", color="gray", ms=7, label="SMART query")
        row.update({"temp_start_C": float(tt.temp.iloc[0]), "temp_end_C": float(tt.temp.iloc[-1]), "temp_max_C": float(tt.temp.max())})
    ax.legend(fontsize=6.5, loc="lower left")
    sust_rows.append(row)
    return s


cliff_rows = []


def cliff_stats(s, label, bytes_per_io):
    """Fast/slow-mode statistics of a sustained-write series (1 s samples): fast = median of the first 60 s; slow second = < 10% of that."""
    if s is None: return
    v = s.val.values.astype(float); fast = float(np.median(v[:60])); slow = v < 0.1 * fast
    row = {"series": label, "seconds": len(v), "fast_median": fast, "pct_seconds_slow": float(100 * slow.mean())}
    if slow.any():
        k = int(np.argmax(slow)); sv = v[k:]; ss = sv < 0.1 * fast
        row.update({"first_slow_s": k, "GB_written_before_first_slow": float(v[:k].sum() * bytes_per_io / 1e9), "pct_slow_after_onset": float(100 * ss.mean()),
                    "slow_median": float(np.median(sv[ss])), "episodes": int(np.sum(np.diff(ss.astype(int)) == 1) + (1 if ss[0] else 0)),
                    "median_episode_s": float(np.median([len(g) for g in np.split(np.arange(len(ss)), np.where(np.diff(ss.astype(int)) != 0)[0] + 1) if ss[g[0]]]))})
    cliff_rows.append(row)


fig, axes = plt.subplots(3, 2, figsize=(13, 12.5))
KIB_TO_MB = 1.024 / 1000                     # FIO bandwidth logs are in KiB/s; the report's MB/s are 10^6 B/s
sb = ts_panel(axes[0, 0], "s9_state__b_sustained_seqwrite128k_qd32", "bw", KIB_TO_MB, C["red"], "throughput (MB/s)", "(a) Sustained sequential 128 KiB writes, QD32, 900 s (stage 9)", "seq 128K write QD32 (MB/s)")
ax = axes[0, 1]
for nm, d, lab, col in (("randread4k", "read", "random read", C["blue"]), ("randwrite4k", "write", "random write", C["orange"])):
    xs, ys, lo, hi = [], [], [], []
    for sz in RANGES:
        mib = int(sz[:-1])
        r = get("s6_range", f"{nm}_qd32_range{sz}", d)
        if r is not None: xs.append(mib); ys.append(r.iops); lo.append(r.iops - r.iops_min); hi.append(r.iops_max - r.iops)
    ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", capsize=2, color=col, label=f"{lab} (QD32)")
ax.set_xscale("log", base=2); ax.set_xlabel("LBA range used (MiB; whole file = 1500 MiB)"); ax.set_ylabel("IOPS"); ax.set_title("(b) Working-set / LBA-range effect (error bars: min-max of 3 trials)", fontsize=9.5); ax.legend(fontsize=7.5)
sr = ts_panel(axes[1, 0], "s9_state__d_sustained_randwrite4k_qd32", "iops", 1, C["orange"], "IOPS (log scale)", "(c) Sustained random 4 KiB writes, QD32, 600 s (stage 9)", "rand 4K write QD32, stage 9 (IOPS)", logy=True)
s11a = ts_panel(axes[1, 1], "s11_cliff__a_randwrite4k_sustained_1200s", "iops", 1, C["orange"], "IOPS (log scale)", "(d) Sustained random 4 KiB writes, QD32, 1200 s (stage 11)", "rand 4K write QD32, stage 11 a (IOPS)", logy=True)
s11b = ts_panel(axes[2, 0], "s11_cliff__b_seqwrite128k_right_after_240s", "bw", KIB_TO_MB, C["red"], "throughput (MB/s)", "(e) Sequential 128 KiB writes right after (d), QD32", "seq 128K write QD32 right after the random-write cliff (MB/s)")
s11c = ts_panel(axes[2, 1], "s11_cliff__c_randwrite4k_after_rest600s_600s", "iops", 1, C["purple"], "IOPS (log scale)", "(f) Random writes again after a 600 s idle rest, 600 s", "rand 4K write QD32 after 600 s rest (IOPS)", logy=True)
for s_, lab in ((sr, "stage 9 d"), (s11a, "stage 11 a"), (s11c, "stage 11 c")): cliff_stats(s_, lab, 4096)
pd.DataFrame(cliff_rows).to_csv(os.path.join(PROC, "write_cliff_summary.csv"), index=False)
fig.suptitle("Project 3 -- Fig. D: device state: sustained writes (burst vs steady state), the random-write cliff, LBA range", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(os.path.join(RES, "figD_device_state.png"), bbox_inches="tight"); plt.close(fig)
# QD=1 latency before / after the heavy writes (table)
states = [("a_fresh", "fresh (before any heavy write)"), ("c_after_seqwrite", "after 900 s sequential writes"), ("e_after_randwrite", "after 600 s random writes"), ("f_after_rest300s", "after a further 300 s idle")]
state_rows = []
with open(os.path.join(GEN, "table_state.tex"), "w") as f:
    f.write(r"\begin{tabular}{lrrrrrrrr}" + "\n" + r"\toprule" + "\n" + r"state & \multicolumn{4}{c}{4 KiB random write, QD=1 ($\mu$s)} & \multicolumn{3}{c}{4 KiB random read, QD=1 ($\mu$s)} \\" + "\n")
    f.write(r" & p50 & p99 & p99.9 & max (ms) & p50 & p99 & p99.9 \\" + "\n" + r"\midrule" + "\n")
    for tag, lab in states:
        w_, r_ = get("s9_state", f"{tag}_randwrite4k_qd1", "write"), get("s9_state", f"{tag}_randread4k_qd1", "read")
        if w_ is None or r_ is None: continue
        f.write(f"{lab} & {w_.p50_us:.0f} & {w_.p99_us:.0f} & {w_.p99_9_us:.0f} & {w_.lat_max_us / 1000:.0f} & {r_.p50_us:.0f} & {r_.p99_us:.0f} & {r_.p99_9_us:.0f} " + r"\\" + "\n")
        state_rows.append({"state": lab, "w_p50": w_.p50_us, "w_p99": w_.p99_us, "w_p999": w_.p99_9_us, "w_max_ms": w_.lat_max_us / 1000, "r_p50": r_.p50_us, "r_p99": r_.p99_us, "r_p999": r_.p99_9_us})
    f.write(r"\bottomrule" + "\n" + r"\end{tabular}" + "\n")
pd.DataFrame(state_rows).to_csv(os.path.join(PROC, "device_state_qd1.csv"), index=False)

# ---- Fig. E: read-throughput stability (sustained QD32 random reads) + misleading setups
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ra = ts_panel(axes[0, 0], "s10_readstab__a_sustained_600s", "iops", 1, C["blue"], "IOPS", "Sustained random 4 KiB reads, QD32: run a", "rand 4K read QD32, run a (IOPS)")
rb = ts_panel(axes[0, 1], "s10_readstab__b_after_rest300s_600s", "iops", 1, C["purple"], "IOPS", "Same after a 300 s idle rest: run b", "rand 4K read QD32, run b after 300 s idle (IOPS)")
pd.DataFrame(sust_rows).to_csv(os.path.join(PROC, "sustained_runs_summary.csv"), index=False)
ax = axes[1, 0]
rd = df[(df.rw == "randread") & (df.bs == "4k") & (df.iodepth >= 16) & (df.dir == "read") & (df.ioengine == "windowsaio") & (df.direct == 1) & (df.numjobs == 1) & (df.ios > 20000)].copy()
if SMART is not None and len(rd):      # temperature vs throughput over every job of the campaign (data only; discussed in the report)
    je = SMART[SMART.tag.str.startswith("job_end:")].dropna(subset=["temperature_c"]).copy(); je["base"] = je.tag.str[8:]
    tmap = je.drop_duplicates("base", keep="last").set_index("base").temperature_c
    rd["temp"] = [tmap.get(f"{s_}__{n_}__t{t_}", np.nan) for s_, n_, t_ in zip(rd.stage, rd.name, rd.trial)]
    rd.dropna(subset=["temp"]).to_csv(os.path.join(PROC, "read_iops_vs_temperature.csv"), index=False)
# time to the first read-throughput step as a function of the LBA range (stage 12): each run starts with a rewrite of the file
s12_rows = []
for nm, lab, col in (("randread4k_qd32_range16m_240s", "16 MiB, QD32", C["red"]), ("randread4k_qd32_range64m_240s", "64 MiB, QD32", C["orange"]), ("randread4k_qd32_range256m_240s", "256 MiB, QD32", C["green"]),
                     ("randread4k_qd32_range1024m_240s", "1024 MiB, QD32", C["blue"]), ("randread4k_qd4_range1024m_600s", "1024 MiB, QD4", C["purple"]),
                     ("randread4k_qd32_range16m_120s_rest600s", "16 MiB, QD32, 600 s rest (120 s run)", C["gray"]), ("randread4k_qd4_range16m_600s", "16 MiB, QD4", C["teal"]),
                     ("randread4k_qd32_range16m_300s_rest600s", "16 MiB, QD32, 600 s rest (300 s run)", "black")):
    s_ = series(f"s12_readcount__{nm}", "iops")
    if s_ is None: continue
    v = s_.val.values.astype(float); t = s_.msec.values / 1000; ref_ = float(np.median(v[:30]))
    ax.plot(t, 100 * v / ref_, color=col, lw=1.1, label=f"{lab} (100% = {ref_ / 1000:.1f} k IOPS)")
    r_ = pd.Series(v).rolling(9, center=True, min_periods=1).median().values
    k = np.where(r_ < 0.9 * ref_)[0]; k = int(k[0]) if len(k) else None
    s12_rows.append({"run": lab, "duration_s": len(v), "reads_total_millions": float(v.sum() / 1e6), "first30s_iops": ref_, "first_step_s": k, "reads_before_step_millions": float(v[:k].sum() / 1e6) if k is not None else float("nan"),
                     "iops_after_step": float(np.median(v[k + 10:k + 60])) if k is not None and len(v) > k + 20 else float("nan"), "min_iops": float(v.min())})
pd.DataFrame(s12_rows).to_csv(os.path.join(PROC, "read_step_vs_range.csv"), index=False)
NL, BSL = chr(10), chr(92)
with open(os.path.join(GEN, "table_readstep.tex"), "w") as f:
    f.write(BSL + "begin{tabular}{lrrrrr}" + NL + BSL + "toprule" + NL + "random 4 KiB reads after a full rewrite of the file & fast rate (k IOPS) & run length (s) & first stall (s) & reads before it (M) & rate after it (k IOPS) " + BSL + BSL + NL + BSL + "midrule" + NL)
    for r in s12_rows:
        st = r["first_step_s"] is not None and not pd.isna(r["first_step_s"])
        f.write(f"{r['run']} & {r['first30s_iops'] / 1000:.1f} & {r['duration_s']} & " + (f"{int(r['first_step_s'])} & {r['reads_before_step_millions']:.1f} & {r['iops_after_step'] / 1000:.1f}" if st else f"none & ({r['reads_total_millions']:.1f} in total) & --") + " " + BSL + BSL + NL)
    f.write(BSL + "bottomrule" + NL + BSL + "end{tabular}" + NL)
ax.set_xlabel("time since start of run (s)"); ax.set_ylabel("% of the first-30-s median"); ax.set_ylim(-3, 108); ax.legend(fontsize=6.5, loc="lower left")
ax.set_title("When does the read ceiling step? After a rewrite: by range, QD and rest (stage 12)", fontsize=9.5)
ax = axes[1, 1]
pit = []
def P(label, stage, name, d="read"):
    r = get(stage, name, d)
    if r is not None: pit.append({"setup": label, "iops": r.iops, "p50": r.p50_us, "p99": r.p99_us})
P("randread QD1\ndirect", "s8_pitfall", "randread4k_qd1_direct"); P("randread QD1 buffered\n(cache mostly cold)", "s8_pitfall", "randread4k_qd1_buffered")
P("randread QD1 buffered\n(cache warm)", "s8_pitfall", "randread4k_qd1_buffered_warm")
P("randread QD32\ndirect", "s8_pitfall", "randread4k_qd32_direct"); P("randread QD32\nbuffered", "s8_pitfall", "randread4k_qd32_buffered")
P("sync engine\n'iodepth=32'\n(buffered)", "s8_pitfall", "randread4k_ioengine_sync_iodepth32_buffered"); P("sync engine\n32 jobs\n(buffered)", "s8_pitfall", "randread4k_ioengine_sync_numjobs32_buffered")
P("windowsaio\n32 jobs x QD1\n(direct)", "s8_pitfall", "randread4k_windowsaio_numjobs32_iodepth1")
P("randwrite QD1\ndirect", "s8_pitfall", "randwrite4k_qd1_direct", "write"); P("randwrite QD1\nbuffered", "s8_pitfall", "randwrite4k_qd1_buffered", "write")
pt = pd.DataFrame(pit); pt.to_csv(os.path.join(PROC, "pitfall_table.csv"), index=False)
cols = [C["blue"], C["red"], C["red"], C["blue"], C["red"], C["red"], C["red"], C["green"], C["blue"], C["red"]]
ax.barh(range(len(pt)), pt.iops, color=cols[:len(pt)]); ax.set_xscale("log"); ax.invert_yaxis()
for i, r in pt.iterrows(): ax.text(r.iops * 1.12, i, f"{r.iops/1000:.1f}k IOPS, p50 {r.p50:.0f} us", va="center", fontsize=6.5)
ax.set_yticks(range(len(pt))); ax.set_yticklabels([s.replace(chr(10), " ") for s in pt.setup], fontsize=6.5); ax.set_xlabel("IOPS (log)"); ax.set_xlim(right=pt.iops.max() * 14)
ax.set_title("Misleading setups: page cache (buffered) and wrong effective queue depth (red = misleading)", fontsize=9)
fig.suptitle("Project 3 -- Fig. E: read-throughput stability and temperature; misleading setups", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figE_read_stability_and_pitfalls.png"), bbox_inches="tight"); plt.close(fig)

# steady-state numbers (sustained writes)
steady = {}
if sb is not None:
    bw = sb.val * KIB_TO_MB
    steady["seq_burst_mbps_first30s"] = float(bw.iloc[:30].median()); steady["seq_last60s_mbps"] = float(bw.iloc[-60:].median()); steady["seq_min_mbps"] = float(bw.min()); steady["seq_total_GB_written"] = float(bw.sum() / 1000)
    drop = np.where(bw.rolling(10).median().values < 0.8 * steady["seq_burst_mbps_first30s"])[0]; steady["seq_drop_t_s"] = float(sb.msec.iloc[drop[0]] / 1000) if len(drop) else float("nan")
if sr is not None:
    steady["rand_first30s_iops"] = float(sr.val.iloc[:30].median()); steady["rand_last60s_iops"] = float(sr.val.iloc[-60:].median()); steady["rand_min_iops"] = float(sr.val.min())
    steady["rand_total_GB_written"] = float(sr.val.sum() * 4096 / 1e9)
pd.Series(steady).to_csv(os.path.join(PROC, "steady_state_summary.csv"), header=["value"])
if len(ref):
    with open(os.path.join(GEN, "table_ref.tex"), "w") as f:
        f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nC++ reference (QD=1, synchronous) & ops/s & mean $\\mu$s & p50 & p99 & p99.9 \\\\\n\\midrule\n")
        for r in refm.itertuples():
            f.write(f"{tex(r.mode)} & {r.iops:,.0f} & {r.mean_us:.1f} & {r.p50_us:.1f} & {r.p99_us:.1f} & {r.p999_us:.1f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

# OS-counter summary of the read-stability runs (host-side saturation check, monitor_os.ps1)
rows = []
for nm in ("a_sustained_600s", "b_after_rest300s_600s"):
    mon = os.path.join(RAW, "logs", f"s10_readstab__{nm}__t0_osmon.csv")
    if not os.path.exists(mon): continue
    m = pd.read_csv(mon); m = m[m["disk reads/sec"] > 1000]
    rows.append({"run": nm, "seconds_with_reads": len(m), "median_disk_reads_per_s": float(m["disk reads/sec"].median()), "median_queue_length": float(m["current disk queue length"].median()),
                 "disk_writes_per_s_max": float(m["disk writes/sec"].max()), "total_cpu_pct_median": float(m["% processor time"].median()),
                 "busiest_cpu_pct_median": float(m["max cpu % processor time"].median()), "busiest_cpu_pct_max": float(m["max cpu % processor time"].max()),
                 "max_interrupt_pct_median": float(m["max cpu % interrupt time"].median()), "max_interrupt_pct_max": float(m["max cpu % interrupt time"].max()),
                 "interrupt_cpu_most_common": str(m["cpu of max % interrupt time"].mode().iloc[0]), "max_dpc_pct_median": float(m["max cpu % dpc time"].median()),
                 "max_dpc_pct_max": float(m["max cpu % dpc time"].max())})
if rows: pd.DataFrame(rows).to_csv(os.path.join(PROC, "os_monitor_summary.csv"), index=False)

# data pattern: compressible vs incompressible payload (s7) -> table
with open(os.path.join(GEN, "table_pattern.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\nworkload (QD=32, write) & payload & IOPS & MB/s & mean $\\mu$s & p99 $\\mu$s \\\\\n\\midrule\n")
    for nm, lab in (("randwrite4k_qd32", "4 KiB random"), ("seqwrite128k_qd32", "128 KiB sequential")):
        for comp, pl in ((0, "incompressible"), (90, "90\\% compressible")):
            r = get("s7_pattern", f"{nm}_compress{comp}", "write")
            if r is not None: f.write(f"{lab if comp == 0 else ''} & {pl} & {r.iops:,.0f} & {r.mbps:.0f} & {r.lat_mean_us:.0f} & {r.p99_us:.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# temperature over the campaign (job_end snapshots) for the report
if SMART is not None:
    je = SMART[SMART.tag.str.startswith("job_end:")].dropna(subset=["temperature_c"]).copy(); je["stage"] = je.tag.str[8:].str.split("__").str[0]
    je.groupby("stage").temperature_c.agg(["min", "median", "max", "count"]).to_csv(os.path.join(PROC, "temperature_by_stage.csv"))
print("done:", PROC, RES, GEN)
