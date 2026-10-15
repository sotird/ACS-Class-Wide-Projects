#!/usr/bin/env python3
"""Aggregate raw per-repetition CSVs -> processed tables + 4 figures + LaTeX tables.

All error bands are the inter-quartile range (25th-75th percentile) of the
per-repetition values (median line); the repetitions are inside one process
run, and run-to-run (process-to-process) variation is reported separately by
scripts/repeat_check.py.

Figures (kept to four, each multi-panel):
  figA  scalar/SSE2/AVX2/AVX-512 throughput per kernel + fine SAXPY locality sweep
  figB  alignment (L1, L2) and tail-handling (ns/call and cycles/element)
  figC  stride/gather (controlled footprint) and data-type comparison
  figD  roofline (L1/L2/L3/DRAM bandwidth roofs, measured FMA peaks) + negative examples
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
COL = {"scalar": "#c44e52", "sse2": "#dd8452", "avx2": "#55a868", "avx512": "#4c72b0", "avx512_fm": "#8172b3", "avx512_gather": "#17becf"}
LAB = {"scalar": "scalar", "sse2": "SSE2 (128b)", "avx2": "AVX2 (256b)", "avx512": "AVX-512 (512b)",
       "avx512_fm": "AVX-512 + -ffast-math", "avx512_gather": "AVX-512, gather forced (use_gather)"}
GIB = 1024.0 ** 3


def load(name, **kw):
    return pd.read_csv(os.path.join(RAW, name), **kw)


def summarize(df, keys, vals):
    g = df.groupby(keys)[vals]
    out = g.median().add_suffix("_med")
    out = out.join(g.quantile(0.25).add_suffix("_q1")).join(g.quantile(0.75).add_suffix("_q3"))
    return out.reset_index()


def tex_escape(s):
    return str(s).replace("_", r"\_").replace("%", r"\%")


# ---- clock: median of all measured samples (start / after_exp1 / end) -------
flog = pd.read_csv(os.path.join(RES, "frequency_log.csv"))
GHZ = float(flog.ghz.median())
GHZ_MIN, GHZ_MAX = float(flog.ghz.min()), float(flog.ghz.max())
print(f"measured clock: median {GHZ:.3f} GHz (min {GHZ_MIN:.3f}, max {GHZ_MAX:.3f})")
with open(os.path.join(PROC, "clock.txt"), "w") as f:
    f.write(f"ghz_median={GHZ:.4f}\nghz_min={GHZ_MIN:.4f}\nghz_max={GHZ_MAX:.4f}\n")

cache = load("cache_topology.csv").drop_duplicates()
L1 = int(cache[(cache.level == 1) & (cache.type == "Data")].size_bytes.max())
L2 = int(cache[cache.level == 2].size_bytes.max())
L3 = int(cache[cache.level == 3].size_bytes.max())

# =============================== exp1 =======================================
BUILDS1 = ["scalar", "sse2", "avx2", "avx512", "avx512_fm"]
e1 = pd.concat([load(f"exp1_{b}.csv") for b in BUILDS1], ignore_index=True)
e1["cycles_per_elem"] = e1.seconds * GHZ * 1e9 / e1.n
e1s = summarize(e1, ["kernel", "dtype", "build", "n"], ["gflops", "gibs", "cycles_per_elem"])
e1s.to_csv(os.path.join(PROC, "exp1_summary.csv"), index=False)

# speedup table (f32) at four representative sizes (x,y f32 footprints ~ 8 B/elem)
SIZES = {"L1": 1 << 12, "L2": 1 << 16, "L3 (eff.)": 1 << 20, "DRAM": 1 << 26}
rows = []
for k in ["saxpy", "dot", "mul", "stencil"]:
    for lvl, n in SIZES.items():
        sub = e1s[(e1s.kernel == k) & (e1s.dtype == "f32") & (e1s.n == n)].set_index("build")
        base = sub.loc["scalar", "gflops_med"]
        rows.append({"kernel": k, "level": lvl, "n": n, "scalar_gflops": base,
                     **{f"{b}_x": sub.loc[b, "gflops_med"] / base for b in BUILDS1[1:]},
                     "avx512_gflops": sub.loc["avx512", "gflops_med"],
                     "scalar_cpe": sub.loc["scalar", "cycles_per_elem_med"],
                     "avx512_cpe": sub.loc["avx512", "cycles_per_elem_med"],
                     "avx512_gibs": sub.loc["avx512", "gibs_med"]})
spd = pd.DataFrame(rows)
spd.to_csv(os.path.join(PROC, "exp1_speedup_table.csv"), index=False)

with open(os.path.join(GEN, "table_speedup.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\n")
    f.write("kernel & level & scalar & SSE2 & AVX2 & AVX-512 & +fast-math & cyc/elem & cyc/elem \\\\\n")
    f.write(" & ($N$) & GFLOP/s & $\\times$ & $\\times$ & $\\times$ & $\\times$ & scalar & AVX-512 \\\\\n\\midrule\n")
    prev = None
    for r in spd.itertuples():
        if prev and prev != r.kernel:
            f.write("\\addlinespace\n")
        prev = r.kernel
        f.write(f"{r.kernel if r.level == 'L1' else ''} & {r.level} ($2^{{{int(np.log2(r.n))}}}$) & {r.scalar_gflops:.2f} & "
                f"{r.sse2_x:.2f} & {r.avx2_x:.2f} & {r.avx512_x:.2f} & {r.avx512_fm_x:.2f} & "
                f"{r.scalar_cpe:.2f} & {r.avx512_cpe:.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---- Figure A -----------------------------------------------------------------
e2 = pd.concat([load(f"exp2_{b}.csv") for b in ["scalar", "avx2", "avx512"]], ignore_index=True)
e2["cycles_per_elem"] = e2.seconds * GHZ * 1e9 / e2.n
e2s = summarize(e2, ["build", "n", "bytes_footprint"], ["gflops", "cycles_per_elem"])
e2s.to_csv(os.path.join(PROC, "exp2_locality_summary.csv"), index=False)

fig = plt.figure(figsize=(13.5, 8.2))
gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.25], hspace=0.42, wspace=0.30)
for i, k in enumerate(["saxpy", "dot", "mul", "stencil"]):
    ax = fig.add_subplot(gs[0, i])
    for b in BUILDS1:
        if b == "avx512_fm" and k != "dot":
            continue
        s = e1s[(e1s.kernel == k) & (e1s.dtype == "f32") & (e1s.build == b)].sort_values("n")
        ls = "--" if b == "avx512_fm" else "-"
        ax.plot(s.n, s.gflops_med, ls, marker="o", ms=2.5, lw=1.2, color=COL[b], label=LAB[b])
        ax.fill_between(s.n, s.gflops_q1, s.gflops_q3, color=COL[b], alpha=0.18, lw=0)
    for cap in (L1, L2, L3):
        ax.axvline(cap / 8, color="gray", ls=":", lw=0.8)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_title(f"{k} (f32)", fontsize=9.5); ax.set_xlabel("N (elements)")
    if i == 0:
        ax.set_ylabel("GFLOP/s (median, IQR band)")
    if i == 1:
        ax.legend(fontsize=6.5, loc="upper right")
axb = fig.add_subplot(gs[1, :])
for b in ["scalar", "avx2", "avx512"]:
    s = e2s[e2s.build == b].sort_values("bytes_footprint")
    axb.plot(s.bytes_footprint, s.gflops_med, "-", lw=1.4, color=COL[b], label=f"{LAB[b]}")
    axb.fill_between(s.bytes_footprint, s.gflops_q1, s.gflops_q3, color=COL[b], alpha=0.2, lw=0)
axb.set_xscale("log"); axb.set_yscale("log")
axb.set_xlabel("Working-set footprint: x + y arrays (bytes, log scale)"); axb.set_ylabel("GFLOP/s")
axb.set_ylim(4, 200)
pv = e2s.pivot(index="bytes_footprint", columns="build", values="gflops_med")
ax2 = axb.twinx()
ax2.plot(pv.index, pv["avx512"] / pv["scalar"], "k--", lw=1.0, label="AVX-512 / scalar speedup")
ax2.plot(pv.index, pv["avx2"] / pv["scalar"], color="#55a868", ls=":", lw=1.0, label="AVX2 / scalar speedup")
ax2.set_ylabel("speedup over scalar (x)"); ax2.set_ylim(0, None); ax2.grid(False)
for cap, name in ((L1, "L1D 48 KiB"), (L2, "L2 1 MiB"), (L3, "L3 32 MiB")):
    axb.axvline(cap, color="gray", ls=":", lw=1)
    axb.text(cap, 190, f" {name}", fontsize=7, color="gray", va="top")
for x0, x1, txt in ((4e3, L1, "L1: compute /\nL1-bandwidth bound"), (L1, L2, "L2-bound"),
                    (L2, L3, "L3-bound"), (L3, 1.2e9, "DRAM-bound: SIMD gain compresses")):
    axb.text(np.sqrt(x0 * x1), 4.5, txt, ha="center", va="bottom", fontsize=7.5, color="#444")
h1, l1_ = axb.get_legend_handles_labels(); h2, l2_ = ax2.get_legend_handles_labels()
axb.legend(h1 + h2, l1_ + l2_, fontsize=7, loc="center right", ncol=1)
axb.set_title("SAXPY f32 locality sweep (fine grid, identical sizes for every build): throughput and SIMD speedup", fontsize=9.5)
fig.suptitle(f"Project 1 -- Fig. A: scalar vs SSE2/AVX2/AVX-512 across kernels and cache levels "
             f"(pinned, {GHZ:.2f} GHz measured)", fontweight="bold", fontsize=11)
fig.savefig(os.path.join(RES, "figA_scalar_vs_vector_and_locality.png"), bbox_inches="tight")
plt.close(fig)

# =============================== exp3 =======================================
B3 = ["scalar", "sse2", "avx2", "avx512"]
e3 = pd.concat([load(f"exp3_{b}.csv") for b in B3], ignore_index=True)
e3["ns_per_call"] = e3.seconds * 1e9
e3["cycles_per_elem"] = e3.seconds * GHZ * 1e9 / e3.n
e3s = summarize(e3, ["sweep", "build", "n", "byte_offset"], ["gflops", "ns_per_call", "cycles_per_elem"])
e3s.to_csv(os.path.join(PROC, "exp3_alignment_tail_summary.csv"), index=False)

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))
for ax, sweep, ttl in ((axes[0, 0], "align_L1", "Alignment, L1-resident (n=2048, 16 KiB)"),
                       (axes[0, 1], "align_L2", "Alignment, L2-resident (n=65536, 512 KiB)")):
    for b in B3:
        s = e3s[(e3s.sweep == sweep) & (e3s.build == b)].sort_values("byte_offset")
        ax.errorbar(s.byte_offset, s.gflops_med, yerr=[s.gflops_med - s.gflops_q1, s.gflops_q3 - s.gflops_med],
                    fmt="o-", ms=3, lw=1.1, capsize=1.5, color=COL[b], label=LAB[b])
    ax.set_xlabel("start offset of x and y from a 64 B boundary (bytes)"); ax.set_ylabel("GFLOP/s"); ax.set_title(ttl, fontsize=9)
    ax.set_yscale("log"); ax.legend(fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0)); ax.set_title(ttl, fontsize=9, pad=22)
for ax, col, yl, ttl in ((axes[1, 0], "ns_per_call", "time per call (ns)", "Tail handling: call time vs n (aligned)"),
                         (axes[1, 1], "cycles_per_elem", "cycles per element", "Tail handling: cycles/element vs n")):
    for b in B3:
        s = e3s[(e3s.sweep == "tail") & (e3s.build == b)].sort_values("n")
        ax.plot(s.n, s[col + "_med"], "-", lw=1.1, color=COL[b], label=LAB[b])
        ax.fill_between(s.n, s[col + "_q1"], s[col + "_q3"], color=COL[b], alpha=0.18, lw=0)
    for m in range(16, 131, 16):
        ax.axvline(m, color="gray", ls=":", lw=0.5)
    ax.set_xlabel("n (elements; dotted = multiples of 16 = one AVX-512 vector)"); ax.set_ylabel(yl); ax.set_title(ttl, fontsize=9)
    if col == "cycles_per_elem": ax.set_yscale("log")
    ax.legend(fontsize=7)
fig.suptitle("Project 1 -- Fig. B: alignment and tail-handling effects on SAXPY f32", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(RES, "figB_alignment_and_tail.png"), bbox_inches="tight")
plt.close(fig)

# tail / alignment table
tail = e3s[e3s.sweep == "tail"]
def tail_stat(b, n): return float(tail[(tail.build == b) & (tail.n == n)].ns_per_call_med.iloc[0])
with open(os.path.join(GEN, "table_tail_align.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n")
    f.write("build & $t(16)$ ns & $t(15)$ ns & $t(17)$ ns & $t(128)$ ns & $t(127)$ ns & misaligned penalty (L1) \\\\\n\\midrule\n")
    for b in B3:
        al = e3s[(e3s.sweep == "align_L1") & (e3s.build == b)].set_index("byte_offset").gflops_med
        pen = 1 - al.drop(0).median() / al.loc[0]
        f.write(f"{LAB[b]} & {tail_stat(b,16):.2f} & {tail_stat(b,15):.2f} & {tail_stat(b,17):.2f} & "
                f"{tail_stat(b,128):.2f} & {tail_stat(b,127):.2f} & {100*pen:.0f}\\% \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# exp3b: is the AVX-512 'tail cost' the masked tail, or a store-to-load dependency between consecutive calls on the same y?
e3b = load("exp3b_tail_rotation.csv")
e3bm = (e3b.groupby(["n", "rotate"]).seconds_per_call.median().unstack() * 1e9).rename(columns={0: "fixed", 1: "indirect", 64: "rot64"})
e3bm.to_csv(os.path.join(PROC, "exp3b_tail_rotation_summary.csv"))
with open(os.path.join(GEN, "table_tail_rotation.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\n$n$ & remainder (lanes) & same $y$ every call (ns) & 64 rotating $y$ buffers (ns) & cycles, rotating & rotating, minus the multiple of 16 below \\\\\n\\midrule\n")
    for n in e3bm.index:
        base = 16 if n == 15 else (n // 16) * 16
        extra = (e3bm.rot64.loc[n] - e3bm.rot64.loc[base]) if base in e3bm.index and base != n else float("nan")
        f.write(f"{n} & {n % 16} & {e3bm.fixed.loc[n]:.2f} & {e3bm.rot64.loc[n]:.2f} & {e3bm.rot64.loc[n] * GHZ:.1f} & {'--' if np.isnan(extra) else format(extra, '+.2f') + ' ns'} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# =============================== exp4 / exp5 ================================
e4 = pd.concat([load("exp4_scalar.csv"), load("exp4_avx512.csv"), load("exp4_avx512_gather.csv")], ignore_index=True)
e4["ns_per_out"] = e4.seconds / e4.n_out * 1e9
e4["cycles_per_out"] = e4.seconds * GHZ * 1e9 / e4.n_out
def line_bytes(p):
    if p == "gather": return 2 * 64 + 4 + 4          # two random lines + z write + idx read
    s = int(p.replace("stride", "")); return 2 * min(64, 4 * s) + 4
e4["useful_gibs"] = 12.0 * e4.n_out / e4.seconds / GIB
e4["line_gibs"] = e4.pattern.map(line_bytes) * e4.n_out / e4.seconds / GIB
e4s = summarize(e4, ["regime", "pattern", "build"], ["ns_per_out", "cycles_per_out", "useful_gibs", "line_gibs"])
e4s.to_csv(os.path.join(PROC, "exp4_stride_gather_summary.csv"), index=False)

B5 = ["scalar", "avx512"]
e5 = pd.concat([load(f"exp5_{b}.csv") for b in B5], ignore_index=True)
e5s = summarize(e5, ["kernel", "dtype", "build", "n"], ["gelem_per_s", "gops"])
e5s.to_csv(os.path.join(PROC, "exp5_dtype_summary.csv"), index=False)

pat_order = ["stride1", "stride2", "stride4", "stride8", "stride16", "gather"]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
for ax, reg in ((axes[0, 0], "L2"), (axes[0, 1], "DRAM")):
    for b in ("scalar", "avx512", "avx512_gather"):
        s = e4s[(e4s.regime == reg) & (e4s.build == b)].set_index("pattern").loc[pat_order]
        x = np.arange(len(pat_order))
        ax.errorbar(x, s.ns_per_out_med, yerr=[s.ns_per_out_med - s.ns_per_out_q1, s.ns_per_out_q3 - s.ns_per_out_med],
                    fmt="o-", color=COL[b], capsize=2, label=LAB[b])
    ax.set_xticks(np.arange(len(pat_order))); ax.set_xticklabels(pat_order, rotation=20)
    ax.set_yscale("log"); ax.set_ylabel("ns per output element")
    ax.set_title(f"{reg}-resident source arrays (identical footprint for every stride)", fontsize=9); ax.legend(fontsize=7.5)
ax = axes[1, 0]
w = 0.2; x = np.arange(len(pat_order))
for j, (reg, b) in enumerate((("L2", "avx512"), ("DRAM", "avx512"))):
    s = e4s[(e4s.regime == reg) & (e4s.build == b)].set_index("pattern").loc[pat_order]
    ax.bar(x + (j - 1.5) * w, s.useful_gibs_med, w, label=f"useful, {reg}, AVX-512", color=["#4c72b0", "#8fa8d0"][j])
    ax.bar(x + (j + 0.5) * w, s.line_gibs_med, w, label=f"cache-line traffic, {reg}, AVX-512", color=["#c44e52", "#e29a9c"][j])
ax.set_xticks(x); ax.set_xticklabels(pat_order, rotation=20); ax.set_yscale("log")
ax.set_ylabel("GiB/s"); ax.set_title("Effective (useful) bandwidth vs bytes of cache lines moved", fontsize=9)
ax.set_ylim(top=ax.get_ylim()[1] * 4); ax.legend(fontsize=6.5, ncol=2, loc="upper right")
ax = axes[1, 1]
sx = e5s[(e5s.kernel == "saxpy")]
ESZ = {"f32": 4, "f64": 8, "i32": 4}
for dt, mk in (("f32", "o"), ("f64", "s"), ("i32", "^")):
    for b, ls in (("avx512", "-"), ("scalar", ":")):
        s = sx[(sx.dtype == dt) & (sx.build == b)].sort_values("n")
        ax.plot(s.n * ESZ[dt] * 2, s.gelem_per_s_med, ls, marker=mk, ms=3, lw=1.1, color={"f32": "#4c72b0", "f64": "#dd8452", "i32": "#55a868"}[dt],
                label=f"{dt} {LAB[b].split(' ')[0]}")
for cap in (L1, L2, L3): ax.axvline(cap, color="gray", ls=":", lw=0.8)
ax.set_xscale("log", base=2); ax.set_yscale("log"); ax.set_xlabel("footprint of x + y arrays (bytes): equal bytes, not equal N"); ax.set_ylabel("G elements / s")
ax.set_title("SAXPY by data type at equal footprint: 16 lanes (f32/i32) vs 8 lanes (f64)", fontsize=9); ax.legend(fontsize=6.5, ncol=2)
fig.suptitle("Project 1 -- Fig. C: stride/gather efficiency (controlled footprint) and data-type comparison", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(RES, "figC_stride_gather_and_dtype.png"), bbox_inches="tight")
plt.close(fig)

# dtype speedup table
rows = []
for k in ("saxpy", "mul", "dot"):
    for dt, lanes in (("f32", 16), ("f64", 8), ("i32", 16)):
        for lvl, n0 in (("L1", 1 << 12), ("L2", 1 << 16), ("DRAM", 1 << 25)):
            n = n0 * 4 // {"f32": 4, "f64": 8, "i32": 4}[dt]          # equal FOOTPRINT: f64 uses half the elements
            sub = e5s[(e5s.kernel == k) & (e5s.dtype == dt) & (e5s.n == n)].set_index("build")
            rows.append({"kernel": k, "dtype": dt, "lanes": lanes, "level": lvl, "n": n,
                         "scalar": sub.loc["scalar", "gelem_per_s_med"], "avx512": sub.loc["avx512", "gelem_per_s_med"],
                         "speedup": sub.loc["avx512", "gelem_per_s_med"] / sub.loc["scalar", "gelem_per_s_med"]})
dts = pd.DataFrame(rows); dts.to_csv(os.path.join(PROC, "exp5_dtype_speedup_table.csv"), index=False)
with open(os.path.join(GEN, "table_dtype.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nkernel & dtype (lanes) & \\multicolumn{2}{c}{L1 (32 KiB of x+y)} & \\multicolumn{2}{c}{DRAM (256 MiB of x+y)} & \\\\\n")
    f.write(" & & AVX-512 Gelem/s & speedup & AVX-512 Gelem/s & speedup & \\\\\n\\midrule\n")
    for k in ("saxpy", "mul", "dot"):
        for dt, lanes in (("f32", 16), ("f64", 8), ("i32", 16)):
            a = dts[(dts.kernel == k) & (dts.dtype == dt) & (dts.level == "L1")].iloc[0]
            d = dts[(dts.kernel == k) & (dts.dtype == dt) & (dts.level == "DRAM")].iloc[0]
            f.write(f"{k if dt=='f32' else ''} & {dt} ({lanes}) & {a.avx512:.2f} & {a.speedup:.1f}$\\times$ & {d.avx512:.2f} & {d.speedup:.1f}$\\times$ & \\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# =============================== roofline ===================================
e7a = load("exp7a_bandwidth.csv")
e7as = summarize(e7a, ["kernel", "n", "footprint_bytes"], ["gibs"])
e7as.to_csv(os.path.join(PROC, "exp7a_bandwidth_summary.csv"), index=False)
def band_ceiling(kernel, lo, hi):
    s = e7as[(e7as.kernel == kernel) & (e7as.footprint_bytes >= lo) & (e7as.footprint_bytes <= hi)]
    return float(s.gibs_med.max())
# fine SAXPY sweep (AVX-512 build), GiB/s derived from gflops: saxpy moves 12 B per 2 flop
sax_gibs = (e2s[e2s.build == "avx512"].assign(gibs=lambda d: d.gflops_med / 2 * 12 / 1.073741824))
def sax_band(lo, hi):
    s_ = sax_gibs[(sax_gibs.bytes_footprint >= lo) & (sax_gibs.bytes_footprint <= hi)]
    return float(s_.gibs.max())
def sax_dram():
    return float(sax_gibs[sax_gibs.bytes_footprint >= 3e8].gibs.median())
# per-level roof = highest bandwidth any of our streaming kernels reaches in that footprint band
BW = {"L1": max(band_ceiling("triad", 6e3, 40e3), sax_band(4e3, 40e3)),
      "L2": max(band_ceiling("triad", 100e3, 700e3), sax_band(100e3, 700e3)),
      "L3": max(band_ceiling("triad", 2e6, 8e6), sax_band(2e6, 8e6)),
      "DRAM": sax_dram()}                                       # in-place read-modify-write (no write-allocate)
BW_DRAM_WA = float(e7as[(e7as.kernel == "triad") & (e7as.footprint_bytes >= 3e8)].gibs_med.median())  # separate output array (write-allocate)
RDBW = {"L1": band_ceiling("sum_u32", 6e3, 40e3), "L2": band_ceiling("sum_u32", 100e3, 700e3),
        "L3": band_ceiling("sum_u32", 2e6, 8e6), "DRAM": float(e7as[(e7as.kernel == "sum_u32") & (e7as.footprint_bytes >= 3e8)].gibs_med.median())}
e7b = load("exp7b_peak_flops.csv")
PEAK = e7b.groupby("isa").gflops.median().to_dict()
FPC = e7b.groupby("isa").flops_per_cycle.median().to_dict()
with open(os.path.join(PROC, "roofline_ceilings.txt"), "w") as f:
    for k, v in BW.items(): f.write(f"triad_bw_gibs_{k}={v:.2f}\n")
    for k, v in RDBW.items(): f.write(f"read_bw_gibs_{k}={v:.2f}\n")
    for k, v in PEAK.items(): f.write(f"peak_gflops_{k}={v:.2f}\nflops_per_cycle_{k}={FPC[k]:.2f}\n")
    f.write(f"ghz={GHZ:.3f}\n")
with open(os.path.join(GEN, "table_ceilings.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrr}\n\\toprule\nlevel & triad GiB/s (12 B/elem) & read-only GiB/s & footprint band \\\\\n\\midrule\n")
    for lvl, band in (("L1", "6--40 KiB"), ("L2", "100--700 KiB"), ("L3", "2--8 MiB"), ("DRAM", "$\\geq$ 300 MB")):
        f.write(f"{lvl} & {BW[lvl]:.1f} & {RDBW[lvl]:.1f} & {band} \\\\\n")
    f.write("\\midrule\nISA & measured GFLOP/s & FLOP/cycle & lanes (f32) \\\\\n\\midrule\n")
    for isa, lanes in (("scalar_fma", 1), ("sse_fma", 4), ("avx2_fma", 8), ("avx512_fma", 16)):
        f.write(f"{tex_escape(isa)} & {PEAK[isa]:.1f} & {FPC[isa]:.1f} & {lanes} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

e8 = load("exp8_negative.csv")
e8["cycles_per_elem"] = e8.seconds * GHZ * 1e9 / e8.n
e8s = summarize(e8, ["variant", "n"], ["gflops", "cycles_per_elem"])
e8s.to_csv(os.path.join(PROC, "exp8_negative_summary.csv"), index=False)

def ai(kernel, n=1 << 20, eb=4):
    w = {"saxpy": (2.0, 3.0), "dot": (2.0, 2.0), "mul": (1.0, 3.0), "stencil": (5.0, 2.0)}[kernel]
    return w[0] / (w[1] * eb)

fig, ax = plt.subplots(figsize=(10.5, 7.2))
x = np.logspace(-2.2, 1.5, 300)
roof_styles = {"L1": ("#9ecae1", "L1 BW"), "L2": ("#6baed6", "L2 BW"), "L3": ("#3182bd", "L3 BW"), "DRAM": ("#08306b", "DRAM BW")}
pk = PEAK["avx512_fma"]
for lvl, (c, lab) in roof_styles.items():
    bw = BW[lvl] * 1.073741824                      # GiB/s -> GB/s
    ax.plot(x, np.minimum(x * bw, pk), color=c, lw=1.6, label=f"{lab} = {BW[lvl]:.0f} GiB/s")
ax.plot(x, np.minimum(x * BW_DRAM_WA * 1.073741824, pk), color="#08306b", lw=1.2, ls="--", label=f"DRAM, write-allocate output = {BW_DRAM_WA:.0f} GiB/s (STREAM-triad convention)")
for isa, c, ls in (("avx512_fma", "black", "-"), ("avx2_fma", "#555555", "--"), ("scalar_fma", "#888888", ":")):
    ax.axhline(PEAK[isa], color=c, ls=ls, lw=1.0)
    ax.text(2.6e1, PEAK[isa] * 1.04, f"{isa.replace('_', ' ')} peak {PEAK[isa]:.0f} GFLOP/s ({FPC[isa]:.0f}/cycle)", fontsize=7.5, ha="right")
mk = {"saxpy": "o", "dot": "s", "mul": "^", "stencil": "D"}
for k in ["saxpy", "dot", "mul", "stencil"]:
    pts = []
    for lvl, n in (("L2", 1 << 14), ("DRAM", 1 << 26)):
        r = e1s[(e1s.kernel == k) & (e1s.dtype == "f32") & (e1s.build == "avx512") & (e1s.n == n)].iloc[0]
        pts.append((ai(k), r.gflops_med))
        ax.plot(ai(k), r.gflops_med, mk[k], ms=8, mfc=COL["avx512"] if lvl == "DRAM" else "white", mec="black", mew=0.9)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="#999999", lw=0.7)
    ax.annotate(k, (ai(k), max(p[1] for p in pts)), xytext=(7, 5), textcoords="offset points", fontsize=8)
r = e1s[(e1s.kernel == "dot") & (e1s.dtype == "f32") & (e1s.build == "avx512_fm") & (e1s.n == 1 << 14)].iloc[0]
ax.plot(ai("dot"), r.gflops_med, "s", ms=9, mfc="#8172b3", mec="black"); ax.annotate("dot + -ffast-math (L2)", (ai("dot"), r.gflops_med), xytext=(8, -3), textcoords="offset points", fontsize=7.5, color="#8172b3")
n8 = int(e8s.n.max()); n8s = int(sorted(e8s.n.unique())[1])
for variant, m, c in (("via_call", "x", "#c44e52"), ("inlined", "+", "#4c72b0")):
    r = e8s[(e8s.variant == variant) & (e8s.n == n8s)].iloc[0]
    ax.plot(2 / 8.0, r.gflops_med, m, ms=11, mew=2.2, color=c)    # y = f(x): read 4B + write 4B, 2 flops
    ax.annotate(f"neg. example: {variant}", (2 / 8.0, r.gflops_med), xytext=(8, -12 if variant == "via_call" else 6), textcoords="offset points", fontsize=7.5, color=c)
ax.plot([], [], "o", mfc="white", mec="black", label="open marker: L2-resident (N=2^14)")
ax.plot([], [], "o", mfc=COL["avx512"], mec="black", label="filled marker: DRAM-resident (N=2^26)")
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(5e-3, 30); ax.set_ylim(0.5, 900)
ax.set_xlabel("arithmetic intensity (FLOP / byte of compulsory traffic, f32)"); ax.set_ylabel("GFLOP/s")
ax.legend(fontsize=7.5, loc="lower right")
ax.set_title(f"Project 1 -- Fig. D: roofline (single core, {GHZ:.2f} GHz): measured bandwidth roofs per level, measured FMA peaks, negative examples",
             fontweight="bold", fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(RES, "figD_roofline_and_negative_example.png"), bbox_inches="tight")
plt.close(fig)

# roofline table: % of the binding roof
rows = []
for k in ["saxpy", "dot", "mul", "stencil"]:
    for lvl, n, roofname in (("L2", 1 << 14, "L2"), ("DRAM", 1 << 26, "DRAM")):
        for b in ("scalar", "avx512", "avx512_fm"):
            if b == "avx512_fm" and k != "dot": continue
            r = e1s[(e1s.kernel == k) & (e1s.dtype == "f32") & (e1s.build == b) & (e1s.n == n)].iloc[0]
            a = ai(k)
            mem_roof = a * BW[roofname] * 1.073741824
            comp_roof = PEAK["avx512_fma"] if b != "scalar" else PEAK["scalar_fma"]
            roof = min(mem_roof, comp_roof)
            # true traffic per element incl. write-allocate reads of output arrays:
            # saxpy 12 B (y is read anyway), dot 8, mul 16 (z: allocate+write), stencil 12 (y: allocate+write)
            true_b = {"saxpy": 12, "dot": 8, "mul": 16, "stencil": 12}[k]
            fl = {"saxpy": 2, "dot": 2, "mul": 1, "stencil": 5}[k]
            true_gibs = r.gflops_med * 1e9 / fl * true_b / GIB
            rows.append({"kernel": k, "level": lvl, "build": b, "ai": a, "gflops": r.gflops_med, "roof": roof,
                         "pct_of_roof": 100 * r.gflops_med / roof, "bound": "memory" if mem_roof < comp_roof else "compute",
                         "true_traffic_gibs": true_gibs, "pct_of_level_bw": 100 * true_gibs / BW[roofname]})
rf = pd.DataFrame(rows); rf.to_csv(os.path.join(PROC, "roofline_points.csv"), index=False)
with open(os.path.join(GEN, "table_roofline.tex"), "w") as f:
    f.write("\\begin{tabular}{llllrrrrrr}\n\\toprule\nkernel & level & build & roof says & intensity & achieved & roof & \\% of roof & true traffic & \\% of level bandwidth \\\\\n")
    f.write(" & & & & FLOP/B & GFLOP/s & GFLOP/s & & GiB/s & \\\\\n\\midrule\n")
    prev = None
    for r in rf.itertuples():
        if prev and prev != r.kernel: f.write("\\addlinespace\n")
        prev = r.kernel
        f.write(f"{r.kernel} & {r.level} & {tex_escape(r.build)} & {r.bound} & {r.ai:.3f} & {r.gflops:.1f} & {r.roof:.1f} & {r.pct_of_roof:.0f}\\% & {r.true_traffic_gibs:.1f} & {r.pct_of_level_bw:.0f}\\% \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

print("processed ->", PROC); print("figures   ->", RES); print("tables    ->", GEN)
