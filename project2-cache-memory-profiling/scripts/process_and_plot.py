#!/usr/bin/env python3
"""Aggregate raw per-repetition CSVs -> processed tables, 4 figures, LaTeX tables.

Error bands/bars are inter-quartile ranges of per-repetition values (median
line). Figures:
  figA  zero-queue latency (+AMAT model) and single-core bandwidth vs footprint
  figB  access pattern / granularity and read/write mix (throughput AND latency)
  figC  loaded-latency curve, knee, Little's Law concurrency
  figD  cache-miss impact (MLP), controlled TLB experiment, benchmark pitfalls
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, PROC, RES = (os.path.join(ROOT, d) for d in ("data/raw", "data/processed", "results"))
GEN = os.path.join(ROOT, "report", "generated")
for d in (PROC, RES, GEN):
    os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
C = {"blue": "#4c72b0", "orange": "#dd8452", "green": "#55a868", "red": "#c44e52", "purple": "#8172b3", "gray": "#777777"}


def load(n): return pd.read_csv(os.path.join(RAW, n))
def summarize(df, keys, vals):
    g = df.groupby(keys)[vals]
    out = g.median().add_suffix("_med").join(g.quantile(0.25).add_suffix("_q1")).join(g.quantile(0.75).add_suffix("_q3"))
    return out.reset_index()
def tex(s): return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
def hum(b):
    b = float(b)
    for u, d in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if b >= d: return f"{b/d:g} {u}"
    return f"{b:g} B"

flog = pd.read_csv(os.path.join(RES, "frequency_log.csv"))
GHZ = float(flog.ghz.median())
with open(os.path.join(PROC, "clock.txt"), "w") as f:
    f.write(f"ghz_median={GHZ:.4f}\nghz_min={flog.ghz.min():.4f}\nghz_max={flog.ghz.max():.4f}\n")
cache = load("cache_topology.csv").drop_duplicates()
L1 = int(cache[(cache.level == 1) & (cache.type == "Data")].size_bytes.max())
L2 = int(cache[cache.level == 2].size_bytes.max())
L3 = int(cache[cache.level == 3].size_bytes.max())

# ================================ exp1 =======================================
e1 = pd.concat([load("exp1_latency_cpu4.csv"), load("exp1_latency_cpu20.csv")], ignore_index=True)
e1s = summarize(e1, ["cpu", "layout", "footprint_bytes"], ["ns_per_hop"])
e1s.to_csv(os.path.join(PROC, "exp1_latency_summary.csv"), index=False)

def plateau(layout, lo, hi, cpu=4):
    s = e1s[(e1s.cpu == cpu) & (e1s.layout == layout) & (e1s.footprint_bytes >= lo) & (e1s.footprint_bytes <= hi)]
    return float(s.ns_per_hop_med.median())
BANDS = {"L1": (4096, 32 * 1024), "L2": (128 * 1024, 768 * 1024), "L3": (2 * 2**20, 8 * 2**20), "DRAM": (512 * 2**20, 2**30)}
lat_rows = []
for lvl, (lo, hi) in BANDS.items():
    rl, pb = plateau("random_lines", lo, hi), plateau("page_blocked", lo, hi)
    lat_rows.append({"level": lvl, "band": f"{hum(lo)}--{hum(hi)}", "random_ns": rl, "pageblocked_ns": pb,
                     "random_cycles": rl * GHZ, "pageblocked_cycles": pb * GHZ,
                     "random_cpu20_ns": plateau("random_lines", lo, hi, 20)})
lat = pd.DataFrame(lat_rows); lat.to_csv(os.path.join(PROC, "zero_queue_latency_table.csv"), index=False)
with open(os.path.join(GEN, "table_latency.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nlevel & footprint band & random ns & page-blocked ns & random cycles & page-blocked cycles & random ns (CPU 20) \\\\\n\\midrule\n")
    for r in lat.itertuples():
        f.write(f"{r.level} & {tex(r.band)} & {r.random_ns:.2f} & {r.pageblocked_ns:.2f} & {r.random_cycles:.0f} & {r.pageblocked_cycles:.0f} & {r.random_cpu20_ns:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# AMAT model on the random-lines curve: h_i = min(1, C_i/F) hit fractions for uniformly random lines
t1, t2, t3, tD = (lat.set_index("level").random_ns[k] for k in ("L1", "L2", "L3", "DRAM"))
def amat(F, C1, C2, C3):
    F = np.asarray(F, float)
    h1, h2, h3 = np.minimum(1, C1 / F), np.minimum(1, C2 / F), np.minimum(1, C3 / F)
    return t1 * h1 + t2 * (h2 - h1) + t3 * (h3 - h2) + tD * (1 - h3)
pb = e1s[(e1s.cpu == 4) & (e1s.layout == "random_lines")].sort_values("footprint_bytes")
grid = np.logspace(np.log2(2 * 2**20), np.log2(256 * 2**20), 120, base=2)
best = min(grid, key=lambda c3: np.mean((np.log(amat(pb.footprint_bytes, L1, L2 + 0, c3)) - np.log(pb.ns_per_hop_med)) ** 2))
C3_FIT = float(best)
with open(os.path.join(PROC, "amat_model.txt"), "w") as f:
    f.write(f"t1={t1:.3f}\nt2={t2:.3f}\nt3={t3:.3f}\ntD={tD:.3f}\nC3_nominal_bytes={L3}\nC3_fitted_bytes={C3_FIT:.0f}\n")

# ================================ exp5 =======================================
e5 = load("exp5_workingset.csv")
e5s = summarize(e5, ["kernel", "footprint_bytes"], ["gbps"])
e5s.to_csv(os.path.join(PROC, "exp5_workingset_summary.csv"), index=False)
def bw_band(kernel, lo, hi):
    s = e5s[(e5s.kernel == kernel) & (e5s.footprint_bytes >= lo) & (e5s.footprint_bytes <= hi)]
    return float(s.gbps_med.max())
BW_BANDS = {"L1": (4096, 40 * 1024), "L2": (100 * 1024, 700 * 1024), "L3": (2 * 2**20, 8 * 2**20), "DRAM": (2**29, 2**31)}
bw_rows = []
for lvl, (lo, hi) in BW_BANDS.items():
    for k in ("read_avx512", "write_avx512", "naive_fp_sum"):
        v = bw_band(k, lo, hi) if lvl != "DRAM" else float(e5s[(e5s.kernel == k) & (e5s.footprint_bytes >= lo)].gbps_med.median())
        bw_rows.append({"level": lvl, "kernel": k, "gbps": v, "bytes_per_cycle": v / GHZ})
bwt = pd.DataFrame(bw_rows); bwt.to_csv(os.path.join(PROC, "bandwidth_levels_table.csv"), index=False)
with open(os.path.join(GEN, "table_bandwidth.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n & \\multicolumn{2}{c}{read (AVX-512)} & \\multicolumn{2}{c}{write (AVX-512)} & \\multicolumn{2}{c}{naive FP sum (pitfall)} \\\\\nlevel & GB/s & B/cycle & GB/s & B/cycle & GB/s & B/cycle \\\\\n\\midrule\n")
    for lvl in BW_BANDS:
        d = bwt[bwt.level == lvl].set_index("kernel")
        f.write(f"{lvl} & {d.gbps['read_avx512']:.1f} & {d.bytes_per_cycle['read_avx512']:.1f} & {d.gbps['write_avx512']:.1f} & {d.bytes_per_cycle['write_avx512']:.1f} & {d.gbps['naive_fp_sum']:.1f} & {d.bytes_per_cycle['naive_fp_sum']:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------- Figure A -------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
ax = axes[0]
for (cpu, layout), col, ls, lab in (((4, "random_lines"), C["blue"], "-", "random lines (CPU 4)"),
                                    ((20, "random_lines"), C["gray"], ":", "random lines (CPU 20, other CCD)"),
                                    ((4, "page_blocked"), C["green"], "-", "page-blocked random (CPU 4): fewer TLB misses, but spatial prefetch helps")):
    s = e1s[(e1s.cpu == cpu) & (e1s.layout == layout)].sort_values("footprint_bytes")
    ax.plot(s.footprint_bytes, s.ns_per_hop_med, ls, color=col, lw=1.4, label=lab)
    ax.fill_between(s.footprint_bytes, s.ns_per_hop_q1, s.ns_per_hop_q3, color=col, alpha=0.2, lw=0)
fx = np.logspace(np.log10(4096), np.log10(2**30), 300)
ax.plot(fx, amat(fx, L1, L2, L3), "k--", lw=1.0, label="AMAT model, nominal capacities (L1 48K, L2 1M, L3 32M)")
ax.plot(fx, amat(fx, L1, L2, C3_FIT), color=C["red"], ls="--", lw=1.0, label=f"AMAT model, fitted effective L3 = {hum(C3_FIT)}")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylabel("latency per access (ns, zero-queue)")
ax.set_title("Zero-queue latency: dependent pointer chase, single outstanding request (median, IQR band)", fontsize=9.5)
ax.legend(fontsize=7, loc="upper left")
txt = "zero-queue plateau, random chase (ns | cycles @ %.2f GHz):\n" % GHZ + "\n".join(
    f"{r.level:5s} {r.random_ns:6.2f} | {r.random_cycles:5.0f}" for r in lat.itertuples())
ax.text(0.985, 0.04, txt, transform=ax.transAxes, fontsize=7.5, family="monospace", ha="right", va="bottom",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
ax = axes[1]
for k, col, lab in (("read_avx512", C["blue"], "read, AVX-512 (int accumulate)"), ("write_avx512", C["orange"], "write, AVX-512 (regular stores)"),
                    ("naive_fp_sum", C["red"], "naive float sum (pitfall: serial FP-add chain)")):
    s = e5s[e5s.kernel == k].sort_values("footprint_bytes")
    ax.plot(s.footprint_bytes, s.gbps_med, color=col, lw=1.4, label=lab)
    ax.fill_between(s.footprint_bytes, s.gbps_q1, s.gbps_q3, color=col, alpha=0.2, lw=0)
ax.set_yscale("log"); ax.set_ylabel("single-core bandwidth (GB/s)")
ax.set_xlabel("working-set footprint (bytes, log scale)")
ax2 = ax.twinx(); ax2.set_yscale("log"); ax2.set_ylim(np.array(ax.get_ylim()) / GHZ); ax2.set_ylabel("bytes / cycle"); ax2.grid(False)
ax.legend(fontsize=7, loc="upper right")
for a in axes:
    for cap, name in ((L1, "L1D"), (L2, "L2"), (L3, "L3")):
        a.axvline(cap, color="gray", ls=":", lw=0.9)
    ytop = a.get_ylim()[1]
    for cap, name in ((L1, "L1D 48K"), (L2, "L2 1M"), (L3, "L3 32M")):
        a.text(cap, ytop, f" {name}", fontsize=7, color="gray", va="top")
fig.suptitle(f"Project 2 -- Fig. A: latency and bandwidth across the cache hierarchy (pinned core, {GHZ:.2f} GHz measured)", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(os.path.join(RES, "figA_latency_and_workingset.png"), bbox_inches="tight"); plt.close(fig)

# ================================ exp2 / exp3 ================================
e2l = summarize(load("exp2_latency.csv"), ["pattern", "param"], ["ns_per_hop"])
e2b = summarize(load("exp2_bandwidth.csv"), ["pattern", "param", "threads"], ["gbps"])
e2l.to_csv(os.path.join(PROC, "exp2_latency_summary.csv"), index=False); e2b.to_csv(os.path.join(PROC, "exp2_bandwidth_summary.csv"), index=False)
e3 = load("exp3_rw_mix.csv")
e3s = summarize(e3, ["mix", "threads"], ["gbps", "lat_p50_ns", "lat_p5_ns", "lat_p95_ns", "lat_mean_ns"])
e3s.to_csv(os.path.join(PROC, "exp3_rw_mix_summary.csv"), index=False)
idle_lat = float(e3s[e3s.mix == "idle"].lat_p50_ns_med.iloc[0])

order = [("stride", 64), ("stride", 256), ("stride", 1024), ("stride", 4096), ("randline", 64),
         ("randblock", 128), ("randblock", 256), ("randblock", 512), ("randblock", 1024), ("randblock", 2048), ("randblock", 4096)]
lab = lambda p, q: {"stride": f"stride\n{q} B", "randline": "random\nline", "randblock": f"rand blk\n{q} B"}[p]
fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
ax = axes[0, 0]
lat_order = [("stride", 64), ("stride", 256), ("stride", 1024), ("stride", 4096), ("randline", 64)]
vals = [e2l[(e2l.pattern == p) & (e2l.param == q)].iloc[0] for p, q in lat_order]
ax.bar(range(len(vals)), [v.ns_per_hop_med for v in vals], color=[C["green"]] * 4 + [C["red"]],
       yerr=[[v.ns_per_hop_med - v.ns_per_hop_q1 for v in vals], [v.ns_per_hop_q3 - v.ns_per_hop_med for v in vals]], capsize=2)
ax.set_xticks(range(len(vals))); ax.set_xticklabels([lab(p, q) for p, q in lat_order], fontsize=7.5)
ax.set_yscale("log"); ax.set_ylabel("ns per dependent access")
for i, v in enumerate(vals): ax.text(i, v.ns_per_hop_med * 1.1, f"{v.ns_per_hop_med:.1f}", ha="center", fontsize=7.5)
ax.set_title("Latency by pattern (1 GiB, single thread, idle machine)", fontsize=9)
ax = axes[0, 1]
w = 0.38; x = np.arange(len(order))
for j, (T, col) in enumerate(((1, C["blue"]), (8, C["orange"]))):
    v = [e2b[(e2b.pattern == p) & (e2b.param == q) & (e2b.threads == T)].iloc[0] for p, q in order]
    ax.bar(x + (j - 0.5) * w, [a.gbps_med for a in v], w, color=col, label=f"{T} thread{'s' if T > 1 else ''}",
           yerr=[[a.gbps_med - a.gbps_q1 for a in v], [a.gbps_q3 - a.gbps_med for a in v]], capsize=1.5)
ax.set_xticks(x); ax.set_xticklabels([lab(p, q) for p, q in order], fontsize=6.5)
ax.set_ylabel("delivered bandwidth (GB/s of 64 B lines)"); ax.set_yscale("log"); ax.legend(fontsize=7.5)
ax.set_title("Bandwidth by pattern / granularity (DRAM-resident)", fontsize=9)
mixes = ["100R", "100W", "100W_NT", "70R30W", "50R50W"]
ax = axes[1, 0]
for j, (T, col) in enumerate(((1, C["blue"]), (4, C["green"]), (14, C["orange"]))):
    v = [e3s[(e3s.mix == m) & (e3s.threads == T)].iloc[0] for m in mixes]
    ax.bar(np.arange(len(mixes)) + (j - 1) * 0.27, [a.gbps_med for a in v], 0.27, color=col, label=f"{T} thread{'s' if T > 1 else ''}",
           yerr=[[a.gbps_med - a.gbps_q1 for a in v], [a.gbps_q3 - a.gbps_med for a in v]], capsize=1.5)
ax.set_xticks(range(len(mixes))); ax.set_xticklabels(mixes); ax.set_ylabel("application throughput (GB/s, loaded + stored)"); ax.legend(fontsize=7.5)
ax.set_title("Read/write mix: throughput", fontsize=9)
ax = axes[1, 1]
for j, (T, col) in enumerate(((1, C["blue"]), (4, C["green"]), (14, C["orange"]))):
    v = [e3s[(e3s.mix == m) & (e3s.threads == T)].iloc[0] for m in mixes]
    xs = np.arange(len(mixes)) + (j - 1) * 0.27
    ax.bar(xs, [a.lat_p50_ns_med for a in v], 0.27, color=col, label=f"{T} thread{'s' if T > 1 else ''}",
           yerr=[[a.lat_p50_ns_med - a.lat_p5_ns_med for a in v], [a.lat_p95_ns_med - a.lat_p50_ns_med for a in v]], capsize=1.5)
ax.axhline(idle_lat, color="k", ls="--", lw=1); ax.text(len(mixes) - 0.5, idle_lat * 1.03, f"idle {idle_lat:.0f} ns", ha="right", fontsize=7.5)
ax.set_xticks(range(len(mixes))); ax.set_xticklabels(mixes); ax.set_ylabel("probe latency under load (ns; whiskers p5--p95)"); ax.legend(fontsize=7.5)
ax.set_title("Read/write mix: loaded latency", fontsize=9)
fig.suptitle("Project 2 -- Fig. B: access pattern / granularity sweep and read/write mix (throughput and latency)", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figB_pattern_granularity_and_rwmix.png"), bbox_inches="tight"); plt.close(fig)

with open(os.path.join(GEN, "table_mix.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nmix & threads & GB/s & probe p50 ns & p5--p95 ns & DRAM bus est. GB/s \\\\\n\\midrule\n")
    for m in mixes:
        for T in (1, 14):
            r = e3s[(e3s.mix == m) & (e3s.threads == T)].iloc[0]
            wfrac = {"100R": 0, "100W": 1, "100W_NT": 1, "70R30W": 0.3, "50R50W": 0.5}[m]
            rfo = 0 if m == "100W_NT" else wfrac            # regular stores read the line first
            bus = r.gbps_med * (1 + rfo)
            f.write(f"{tex(m)} & {T} & {r.gbps_med:.1f} & {r.lat_p50_ns_med:.0f} & {r.lat_p5_ns_med:.0f}--{r.lat_p95_ns_med:.0f} & {bus:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ================================ exp4 =======================================
e4 = load("exp4_intensity.csv")
e4s = summarize(e4, ["mix", "sweep", "threads", "delay_us"], ["gbps", "lat_p50_ns", "lat_p5_ns", "lat_p95_ns"])
e4s["outstanding_lines"] = e4s.gbps_med * 1e9 * e4s.lat_p50_ns_med * 1e-9 / 64
e4s.to_csv(os.path.join(PROC, "exp4_intensity_summary.csv"), index=False)
knees = []
fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
for mix, col in (("100R", C["blue"]), ("70R30W", C["orange"])):
    d = e4s[(e4s.mix == mix) & (e4s.sweep == "delay")].sort_values("gbps_med")
    axes[0].plot(d.gbps_med, d.lat_p50_ns_med, "o-", ms=4, color=col, lw=1.3, label=f"{mix}: 14 threads, delay swept")
    axes[0].fill_between(d.gbps_med, d.lat_p5_ns_med, d.lat_p95_ns_med, color=col, alpha=0.15, lw=0)
    th = e4s[(e4s.mix == mix) & (e4s.sweep == "threads")].sort_values("gbps_med")
    axes[0].plot(th.gbps_med, th.lat_p50_ns_med, "s--", ms=3.5, color=col, lw=0.8, alpha=0.7, label=f"{mix}: delay 0, threads swept")
    peak = float(d.gbps_med.max()); sat = d[d.gbps_med >= 0.97 * peak]
    peak_lat = float(sat.lat_p50_ns_med.median()); base = idle_lat
    def cross(thr):                         # bandwidth where latency first reaches thr (linear interpolation along the sweep)
        b, l = d.gbps_med.values, d.lat_p50_ns_med.values
        for i in range(1, len(b)):
            if l[i - 1] < thr <= l[i]:
                return float(b[i - 1] + (thr - l[i - 1]) / (l[i] - l[i - 1]) * (b[i] - b[i - 1]))
        return float("nan")
    k125, k150 = cross(1.25 * base), cross(1.5 * base)
    knees.append({"mix": mix, "unloaded_ns": base, "knee125_gbps": k125, "knee125_pct": 100 * k125 / peak, "knee150_gbps": k150, "knee150_pct": 100 * k150 / peak,
                  "peak_gbps": peak, "peak_lat_ns": peak_lat, "lat_inflation_at_peak": peak_lat / base,
                  "N_at_knee125": k125 * 1e9 * 1.25 * base * 1e-9 / 64, "N_at_peak": peak * 1e9 * peak_lat * 1e-9 / 64})
    axes[0].plot(k125, 1.25 * base, "*", ms=15, color="k", zorder=5)
    axes[0].annotate(f"knee ({mix}): latency = 1.25x idle\n{k125:.0f} GB/s = {100*k125/peak:.0f}% of peak", (k125, 1.25 * base),
                     xytext=(-110, 38) if mix == "70R30W" else (15, -45), textcoords="offset points",
                     fontsize=7.5, arrowprops=dict(arrowstyle="-", color="gray"))
axes[0].axvline(96.0, color="gray", ls=":", lw=1); axes[0].text(95, axes[0].get_ylim()[1] * 0.97, "DDR5-6000 x2 theoretical 96 GB/s ", ha="right", va="top", fontsize=7.5, color="gray")
axes[0].set_xlabel("delivered bandwidth (GB/s)"); axes[0].set_ylabel("probe latency, median (ns; band p5--p95)")
axes[0].set_title("Throughput-latency curve (loaded latency)", fontsize=9.5); axes[0].legend(fontsize=7, loc="upper left")
for mix, col in (("100R", C["blue"]), ("70R30W", C["orange"])):
    d = e4s[(e4s.mix == mix) & (e4s.sweep == "delay")].sort_values("gbps_med")
    axes[1].plot(d.gbps_med, d.outstanding_lines, "o-", ms=4, color=col, label=f"{mix}")
axes[1].set_xlabel("delivered bandwidth (GB/s)"); axes[1].set_ylabel("N = bandwidth x latency / 64 B  (lines in flight)")
axes[1].set_title("Little's Law: implied memory-level concurrency", fontsize=9.5); axes[1].legend(fontsize=8)
fig.suptitle("Project 2 -- Fig. C: loaded-latency curve, knee, and Little's Law", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(os.path.join(RES, "figC_loaded_latency_curve.png"), bbox_inches="tight"); plt.close(fig)
kn = pd.DataFrame(knees); kn.to_csv(os.path.join(PROC, "knee_table.csv"), index=False)
with open(os.path.join(GEN, "table_knee.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrrrr}\n\\toprule\nmix & idle ns & BW at 1.25$\\times$ idle (knee) & BW at 1.5$\\times$ & peak GB/s & ns at peak & latency $\\times$ & $N$ at knee & $N$ at peak \\\\\n\\midrule\n")
    for r in kn.itertuples():
        f.write(f"{tex(r.mix)} & {r.unloaded_ns:.0f} & {r.knee125_gbps:.1f} ({r.knee125_pct:.0f}\\%) & {r.knee150_gbps:.1f} ({r.knee150_pct:.0f}\\%) & {r.peak_gbps:.1f} & {r.peak_lat_ns:.0f} & {r.lat_inflation_at_peak:.2f} & {r.N_at_knee125:.0f} & {r.N_at_peak:.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ================================ exp6 / exp7 / exp8 ==========================
e6 = load("exp6_cachemiss.csv"); e6s = summarize(e6, ["pattern", "footprint_bytes"], ["ns_per_line"])
e6s.to_csv(os.path.join(PROC, "exp6_cachemiss_summary.csv"), index=False)
K4, K2 = "4K", "2M"                                    # page kinds in the exp7 CSVs: 4 KiB pages, 2 MiB huge pages
e7 = load("exp7_tlb.csv"); e7s = summarize(e7, ["L_lines", "P_pages", "page_kind"], ["ns_per_hop"])
e7s.to_csv(os.path.join(PROC, "exp7_tlb_summary.csv"), index=False)
huge = bool((e7.page_kind == K2).any())
# the first huge-page run used the address-order layout (page number correlated with line offset): kept as the layout-pitfall exhibit
e7u = load("exp7_tlb_unpermuted_layout.csv"); e7us = summarize(e7u, ["L_lines", "P_pages", "page_kind"], ["ns_per_hop"])
e7w = load("exp7_hugepage_wsweep.csv"); e7ws = summarize(e7w, ["footprint_bytes", "page_kind"], ["ns_per_hop"])
e7ws.to_csv(os.path.join(PROC, "exp7_hugepage_wsweep_summary.csv"), index=False)
e8a = load("exp8_cache_reuse.csv"); e8b = load("exp8_pinning.csv"); e8c = load("exp8_first_touch.csv"); e8d = load("exp8_address_generation.csv")

fig, axes = plt.subplots(3, 2, figsize=(13, 13.5))
ax = axes[0, 0]
for p, col, lab_ in (("seq_indep", C["green"], "sequential, independent loads"), ("rand_indep", C["orange"], "random, independent loads (MLP)"),
                     ("rand_dep", C["red"], "random, dependent chase (serialized)")):
    s = e6s[e6s.pattern == p].sort_values("footprint_bytes")
    ax.plot(s.footprint_bytes, s.ns_per_line_med, "o-", ms=3, color=col, label=lab_)
    ax.fill_between(s.footprint_bytes, s.ns_per_line_q1, s.ns_per_line_q3, color=col, alpha=0.2, lw=0)
for cap in (L1, L2, L3): ax.axvline(cap, color="gray", ls=":", lw=0.9)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("footprint (bytes)"); ax.set_ylabel("ns per 64 B line")
ax.set_title("Cache-miss impact: same lines, three access disciplines", fontsize=9); ax.legend(fontsize=7, loc="upper left")
dep = e6s[e6s.pattern == "rand_dep"].set_index("footprint_bytes").ns_per_line_med; ind = e6s[e6s.pattern == "rand_indep"].set_index("footprint_bytes").ns_per_line_med
mlp = (dep / ind).dropna(); axm = ax.twinx(); axm.plot(mlp.index, mlp.values, "k:", lw=1); axm.set_ylabel("achieved MLP = dependent / independent (dotted)"); axm.grid(False)
ax = axes[0, 1]
for L, col in ((512, C["green"]), (8192, C["blue"]), (65536, C["red"])):
    for kind, ls, mk, lab in ((K4, "-", "o", "4 KiB pages"), (K2, "--", "s", "2 MiB pages")):
        s = e7s[(e7s.L_lines == L) & (e7s.page_kind == kind)].sort_values("P_pages")
        if s.empty: continue
        ax.plot(s.P_pages, s.ns_per_hop_med, ls, marker=mk, ms=4, color=col, label=f"{hum(L*64)} of lines, {lab}")
        ax.fill_between(s.P_pages, s.ns_per_hop_q1, s.ns_per_hop_q3, color=col, alpha=0.2, lw=0)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("distinct 4 KiB-page slots touched (same cache lines spread over P slots)")
ax.set_ylabel("ns per dependent access"); ax.legend(fontsize=6.5, loc="upper left", ncol=1)
ax.set_title("TLB-miss impact at constant cache footprint: 4 KiB pages (solid) vs 2 MiB pages (dashed)", fontsize=9)

# huge pages and the working set: a dependent chase over fully random lines, 128 KiB ... 1 GiB
ax = axes[1, 0]
for kind, col, ls, mk, lab in ((K4, C["red"], "-", "o", "4 KiB pages"), (K2, C["green"], "--", "s", "2 MiB pages")):
    s = e7ws[e7ws.page_kind == kind].sort_values("footprint_bytes")
    ax.plot(s.footprint_bytes, s.ns_per_hop_med, ls, marker=mk, ms=4, color=col, label=lab)
    ax.fill_between(s.footprint_bytes, s.ns_per_hop_q1, s.ns_per_hop_q3, color=col, alpha=0.2, lw=0)
for cap in (L1, L2, L3): ax.axvline(cap, color="gray", ls=":", lw=0.9)
w4 = e7ws[e7ws.page_kind == K4].set_index("footprint_bytes").ns_per_hop_med; w2 = e7ws[e7ws.page_kind == K2].set_index("footprint_bytes").ns_per_hop_med
axg = ax.twinx(); axg.plot(w4.index, 100 * (w4 - w2) / w4, "k:", marker=".", lw=1); axg.set_ylabel("latency saved by huge pages (%, dotted)"); axg.grid(False); axg.set_ylim(0, 40)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("footprint (bytes)"); ax.set_ylabel("ns per dependent access (random lines)")
ax.legend(fontsize=7.5, loc="upper left"); ax.set_title("Huge pages vs. working set: how much of a miss is page-walk cost", fontsize=9)

# layout pitfall on huge pages: with 2 MiB pages physical = virtual, so the layout itself chooses the cache sets
ax = axes[1, 1]
for kind, d, col, ls, mk, lab in ((K4, e7s, C["blue"], "-", "o", "4 KiB pages (OS-random frames)"),
                                  (K2, e7us, C["red"], "--", "s", "2 MiB pages, address-order layout (set conflicts)"),
                                  (K2, e7s, C["green"], "--", "s", "2 MiB pages, page slots permuted (fixed)")):
    s = d[(d.L_lines == 8192) & (d.page_kind == kind)].sort_values("P_pages")
    ax.plot(s.P_pages, s.ns_per_hop_med, ls, marker=mk, ms=4, color=col, label=lab)
ax.set_xscale("log", base=2); ax.set_xlabel("distinct 4 KiB-page slots (512 KiB of lines, L2-resident)"); ax.set_ylabel("ns per dependent access")
ax.legend(fontsize=7, loc="upper left"); ax.set_title("Pitfall: on huge pages the layout, not the OS, picks the cache sets", fontsize=9)
ax = axes[2, 0]
a = e8a.groupby("variant")[["claimed_gbps", "unique_bytes_gbps"]].median()
ff = e8c.groupby("state").gbps.median()
ag = e8d.groupby("variant").gbps.median()
labels = ["buggy:\nclaimed", "buggy:\nunique bytes/s", "fixed:\nclaimed", "1st pass\n(untouched)", "2nd pass\n(same mem)",
          "rand lines:\nprecomputed\naddresses", "rand lines:\nPRNG in loop"]
vals = [a.claimed_gbps["buggy_masked_to_256KiB"], a.unique_bytes_gbps["buggy_masked_to_256KiB"], a.claimed_gbps["fixed_full_512MiB"],
        ff["first_pass_untouched"], ff["second_pass"], ag["precomputed_with_replacement"], ag["inline_xorshift_in_loop"]]
ax.bar(range(7), vals, color=[C["orange"], C["green"], C["blue"], C["red"], C["blue"], C["blue"], C["red"]]); ax.set_yscale("log")
for i, v in enumerate(vals): ax.text(i, v * 1.1, f"{v:.2g}", ha="center", fontsize=8)
ax.set_xticks(range(7)); ax.set_xticklabels(labels, fontsize=6.8); ax.set_ylabel("GB/s (log)")
ax.set_title("Pitfalls: cache reuse as DRAM bandwidth; first-touch faults; address generation in the loop", fontsize=9)
ax = axes[2, 1]
conds = ["pinned_idle", "unpinned_idle", "pinned_loaded", "unpinned_loaded"]
data = [e8b[e8b.condition == c].ns_per_hop.values for c in conds]
bp = ax.boxplot(data, tick_labels=[c.replace("_", "\n") for c in conds], showfliers=True, patch_artist=True, widths=0.55)
for patch, c in zip(bp["boxes"], [C["green"], C["orange"], C["green"], C["orange"]]): patch.set_facecolor(c); patch.set_alpha(0.6)
for i, c in enumerate(conds, 1):
    ncpu = e8b[e8b.condition == c].cpu.nunique(); cv = 100 * e8b[e8b.condition == c].ns_per_hop.std() / e8b[e8b.condition == c].ns_per_hop.mean()
    ax.text(i, ax.get_ylim()[1], f"CPUs: {ncpu}\nCV {cv:.1f}%", ha="center", va="top", fontsize=7.5)
ax.set_ylabel("ns/hop per 25 ms window (DRAM-resident chase)"); ax.set_title("Pitfall: not pinning / no isolation (60 windows each)", fontsize=9)
fig.suptitle("Project 2 -- Fig. D: cache-miss impact, TLB-miss impact with 4 KiB and 2 MiB pages, and benchmark pitfalls", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figD_cachemiss_tlbmiss_and_pitfall.png"), bbox_inches="tight"); plt.close(fig)

# TLB / pitfall tables
tl = []
for L in (512, 8192, 65536):
    row = {"L_lines": L}
    for kind in (K4, K2):
        s = e7s[(e7s.L_lines == L) & (e7s.page_kind == kind)].sort_values("P_pages")
        if s.empty: continue
        row.update({f"first_{kind}": float(s.ns_per_hop_med.iloc[0]), f"last_{kind}": float(s.ns_per_hop_med.iloc[-1]),
                    f"penalty_{kind}": float(s.ns_per_hop_med.iloc[-1] - s.ns_per_hop_med.iloc[0]),
                    "P_first": int(s.P_pages.iloc[0]), "P_last": int(s.P_pages.iloc[-1])})
    tl.append(row)
tlt = pd.DataFrame(tl); tlt.to_csv(os.path.join(PROC, "tlb_table.csv"), index=False)
with open(os.path.join(GEN, "table_tlb.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrrrr}\n\\toprule\n & & \\multicolumn{3}{c}{4\\,KiB pages} & \\multicolumn{3}{c}{2\\,MiB pages} \\\\\n"
            "\\cmidrule(lr){3-5}\\cmidrule(lr){6-8}\ncache footprint & pages ($P$) range & ns, fewest & ns, most & penalty & ns, fewest & ns, most & penalty \\\\\n\\midrule\n")
    for r in tlt.itertuples():
        f.write(f"{hum(r.L_lines*64)} & {r.P_first}--{r.P_last} & {r.first_4K:.2f} & {r.last_4K:.2f} & {r.penalty_4K:+.2f} & {r.first_2M:.2f} & {r.last_2M:.2f} & {r.penalty_2M:+.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# huge pages vs working set (random-line dependent chase)
hw = pd.DataFrame({"ns_4K": w4, "ns_2M": w2}); hw["saved_ns"] = hw.ns_4K - hw.ns_2M; hw["saved_pct"] = 100 * hw.saved_ns / hw.ns_4K
hw["spread_4K_pct"] = 100 * (e7ws[e7ws.page_kind == K4].set_index("footprint_bytes").ns_per_hop_q3 - e7ws[e7ws.page_kind == K4].set_index("footprint_bytes").ns_per_hop_q1) / hw.ns_4K
hw.to_csv(os.path.join(PROC, "hugepage_wsweep_table.csv"))
with open(os.path.join(GEN, "table_hugepages_ws.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrr}\n\\toprule\nfootprint (random lines) & ns/access, 4\\,KiB pages & ns/access, 2\\,MiB pages & saved (ns) & saved (\\%) \\\\\n\\midrule\n")
    for fp, r in hw.iterrows():
        f.write(f"{hum(fp)} & {r.ns_4K:.2f} & {r.ns_2M:.2f} & {r.saved_ns:.2f} & {r.saved_pct:.1f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# layout pitfall table: 512 KiB of lines, 2 MiB pages, address-order vs permuted slots (worst cases)
lp = []
for P in sorted(e7us[(e7us.L_lines == 8192)].P_pages.unique()):
    g = lambda d, k: float(d[(d.L_lines == 8192) & (d.P_pages == P) & (d.page_kind == k)].ns_per_hop_med.iloc[0])
    lp.append({"P_pages": P, "ns_4K": g(e7s, K4), "ns_2M_addrorder": g(e7us, K2), "ns_2M_permuted": g(e7s, K2)})
pd.DataFrame(lp).to_csv(os.path.join(PROC, "hugepage_layout_pitfall.csv"), index=False)
pr = e8b.groupby("condition").ns_per_hop.agg(["median", "mean", "std", "min", "max"]); pr["cv_pct"] = 100 * pr["std"] / pr["mean"]; pr["cpus"] = e8b.groupby("condition").cpu.nunique()
pr.to_csv(os.path.join(PROC, "pinning_table.csv"))
with open(os.path.join(GEN, "table_pinning.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrr}\n\\toprule\ncondition & median ns/hop & max ns/hop & CV & CPUs visited \\\\\n\\midrule\n")
    for c in conds:
        r = pr.loc[c]; f.write(f"{tex(c)} & {r['median']:.1f} & {r['max']:.1f} & {r.cv_pct:.1f}\\% & {int(r.cpus)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
print("processed ->", PROC); print("tables ->", GEN)
