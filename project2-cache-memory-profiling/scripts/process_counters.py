#!/usr/bin/env python3
"""Hardware-counter evidence for Project 2 (exp9): raw counter CSV -> figE_counters.png, processed CSVs, LaTeX tables.

Input : data/raw/counters/exp9_counters.csv  (scripts/run_counters.ps1; counts per unit = hop / line / element, setup differenced out)
Output: results/figE_counters.png, data/processed/counters_*.csv, report/generated/table_counters_*.tex

Counters (AMD Zen 5 PMCs read through ETW): TotalCycles, InstructionRetired, FillL2 / FillL3 / FillDram = L1D fills from L2 / L3 / DRAM
(event 0x44 LsAnyFillsFromSys, source masks 0x01 / 0x02 / 0x48), DtlbMiss = L1 DTLB misses (event 0x45, mask 0xFF).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW, PROC, RES = (os.path.join(ROOT, d) for d in ("data/raw/counters", "data/processed", "results"))
GEN = os.path.join(ROOT, "report", "generated")
for d in (PROC, RES, GEN): os.makedirs(d, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 8.5, "axes.grid": True, "grid.alpha": 0.3,
                     "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
C = {"blue": "#4c72b0", "orange": "#dd8452", "green": "#55a868", "red": "#c44e52", "purple": "#8172b3", "gray": "#777777"}
raw = pd.read_csv(os.path.join(RAW, "exp9_counters.csv"))
raw.to_csv(os.path.join(PROC, "counters_all_repetitions.csv"), index=False)
num = [c for c in raw.columns if c.endswith("_per_unit")]
df = raw.groupby(["exp", "label"], sort=False)[num].median().reset_index()      # median over the repetitions of a configuration
df["reps"] = raw.groupby(["exp", "label"], sort=False).size().values
df = df.rename(columns={"TotalCycles_per_unit": "cyc", "InstructionRetired_per_unit": "instr", "FillL2_per_unit": "fL2", "FillL3_per_unit": "fL3",
                        "FillDram_per_unit": "fD", "DtlbMiss_per_unit": "dtlb", "ns_per_unit": "ns"})
GHZ = float(open(os.path.join(ROOT, "data", "processed", "clock.txt")).read().split("ghz_median=")[1].split()[0])
def tex(s): return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
L1, L2, L3 = 48, 1024, 32768          # KiB (Zen 5: 48 KiB L1D, 1 MiB L2, 32 MiB L3 per CCD)
Z4 = lambda x: f"{0.0 if abs(x) < 5e-4 else x:.4f}"      # per-element fills: noise of a few 1e-4 is not signal
def Z(x, d=3):
    """format a counter ratio; differenced counters can come out as -0.0003 when the true value is 0"""
    return f"{0.0 if abs(x) < 0.5 * 10 ** -d else x:.{d}f}"

# ---------------------------------------------------------------- E1: working-set sweep (compact pages)
e1 = df[df.exp == "E1_chase"].copy(); e1["kib"] = e1.label.str.split().str[0].astype(int); e1 = e1.sort_values("kib")
# ---------------------------------------------------------------- E3: TLB reach at fixed cache footprint
e3 = df[df.exp == "E3_tlb"].copy()
e3["kib"] = e3.label.str.split().str[0].astype(int); e3["pages"] = e3.label.str.split().str[3].astype(int)
# ---------------------------------------------------------------- latency model from counters: ns = c0 + cL2*fL2 + cL3*fL3 + cD*fD + cT*dtlb
fit = pd.concat([e1, e3])
Xcols = ["fL2", "fL3", "fD", "dtlb"]
A = np.column_stack([np.ones(len(fit))] + [fit[c].values for c in Xcols]); y = fit.ns.values
W = 1.0 / y                                   # weighted least squares = minimise RELATIVE error (an unweighted fit is dominated by the 90 ns DRAM points)
coef, *_ = np.linalg.lstsq(A * W[:, None], y * W, rcond=None)
pred = A @ coef
relerr = np.abs(pred - y) / y
r2 = 1 - ((np.log(y) - np.log(pred)) ** 2).sum() / ((np.log(y) - np.log(y).mean()) ** 2).sum()      # R^2 on the log scale (all levels count equally)
MEDERR, MAXERR = 100 * float(np.median(relerr)), 100 * float(relerr.max())
fit["pred"] = pred
names = ["base: L1 hit (load-to-use)", "extra per L1D fill from L2", "extra per L1D fill from L3", "extra per L1D fill from DRAM", "extra per DTLB miss"]
pd.DataFrame({"term": names, "ns": coef}).to_csv(os.path.join(PROC, "counters_latency_fit.csv"), index=False)
with open(os.path.join(GEN, "table_counters_fit.tex"), "w") as f:
    f.write("\\begin{tabular}{lrr}\n\\toprule\nterm of the model $t = t_0 + \\sum_i n_i c_i$ & fitted $c_i$ (ns) & cycles at %.2f\\,GHz \\\\\n\\midrule\n" % GHZ)
    for n_, c_ in zip(names, coef): f.write(f"{n_} & {c_:.2f} & {c_ * GHZ:.0f} \\\\\n")
    f.write(f"\\bottomrule\n\\multicolumn{{3}}{{l}}{{\\footnotesize weighted least squares over {len(fit)} configurations: median error {MEDERR:.0f}\\%, maximum {MAXERR:.0f}\\%, $R^2$ (log scale) $={r2:.3f}$}}\\\\\n\\end{{tabular}}\n")
# E1 table
with open(os.path.join(GEN, "table_counters_chase.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrrrr}\n\\toprule\nfootprint & ns/hop & cycles & instr. & fills from L2 & from L3 & from DRAM & DTLB miss \\\\\n & & per hop & per hop & per hop & per hop & per hop & per hop \\\\\n\\midrule\n")
    for r in e1.itertuples():
        f.write(f"{r.label} & {r.ns:.2f} & {r.cyc:.1f} & {r.instr:.2f} & {Z(r.fL2)} & {Z(r.fL3)} & {Z(r.fD)} & {Z(r.dtlb)} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
e1.to_csv(os.path.join(PROC, "counters_chase.csv"), index=False)
# E2 disciplines
e2 = df[df.exp == "E2_discipline"].copy(); e2.to_csv(os.path.join(PROC, "counters_discipline.csv"), index=False)
dep_ns = float(e2[e2.label == "dependent chase"].ns.iloc[0]) if len(e2) else np.nan
with open(os.path.join(GEN, "table_counters_discipline.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrrr}\n\\toprule\n1 GiB of lines, touched\\ldots & ns/line & cycles & instr. & DRAM fills/line & DTLB miss/line & implied MLP \\\\\n\\midrule\n")
    for r in e2.itertuples():
        f.write(f"{tex(r.label)} & {r.ns:.2f} & {r.cyc:.1f} & {r.instr:.2f} & {Z(r.fD)} & {Z(r.dtlb)} & {dep_ns / r.ns:.0f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# E3 table
with open(os.path.join(GEN, "table_counters_tlb.tex"), "w") as f:
    f.write("\\begin{tabular}{rrrrrr}\n\\toprule\ncache footprint & pages & ns/hop & DTLB miss/hop & fills from L2/L3/DRAM per hop & penalty vs.\\ fewest pages (ns) \\\\\n\\midrule\n")
    for kib, g in e3.groupby("kib"):
        g = g.sort_values("pages"); base = g.ns.iloc[0]
        for r in g.itertuples():
            f.write(f"{kib} KiB & {r.pages} & {r.ns:.2f} & {Z(r.dtlb)} & {Z(r.fL2, 2)} / {Z(r.fL3, 2)} / {Z(r.fD, 2)} & {Z(r.ns - base, 2).replace("0.00", "+0.00") if abs(r.ns - base) < 0.005 else format(r.ns - base, "+.2f")} \\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
e3.to_csv(os.path.join(PROC, "counters_tlb.csv"), index=False)
# E5: 4 KiB vs 2 MiB pages, identical layout and randomly permuted page slots (needs SeLockMemoryPrivilege; rows exist only if run_counters.ps1 -Only E5_hugepages was run)
e5 = df[df.exp == "E5_hugepages"].copy()
if len(e5):
    e5["kib"] = e5.label.str.split().str[0].astype(int); e5["pages"] = e5.label.str.split().str[3].str.rstrip(",").astype(int)
    e5["psize"] = np.where(e5.label.str.contains("2 MiB pages"), "2 MiB", "4 KiB")
    e5 = e5.sort_values(["kib", "pages", "psize"])
    # out-of-sample test of the latency model fitted on E1 + E3 only (4 KiB pages, address-order layout): does it predict the new rows?
    e5["pred"] = coef[0] + e5.fL2.clip(lower=0) * coef[1] + e5.fL3.clip(lower=0) * coef[2] + e5.fD.clip(lower=0) * coef[3] + e5.dtlb.clip(lower=0) * coef[4]
    e5["relerr_pct"] = 100 * (e5.pred - e5.ns) / e5.ns
    print(f"E5 out-of-sample latency-model error: median |err| {e5.relerr_pct.abs().median():.1f}% (by page size: "
          f"{e5.groupby('psize').relerr_pct.apply(lambda s: round(float(s.abs().median()), 1)).to_dict()}), max |err| {e5.relerr_pct.abs().max():.1f}%")
    with open(os.path.join(GEN, "table_counters_hugepages.tex"), "w") as f:
        f.write("\\begin{tabular}{rrlrrrrr}\n\\toprule\ncache footprint & page slots & page size & ns/hop & cycles/hop & DTLB miss/hop & fills from L2 / L3 per hop & runs \\\\\n\\midrule\n")
        for (kib, pages), g in e5.groupby(["kib", "pages"], sort=True):
            for r in g.itertuples():
                f.write(f"{kib} KiB & {pages} & {r.psize} & {r.ns:.2f} & {r.cyc:.1f} & {Z(max(r.dtlb, 0))} & {Z(r.fL2, 2)} / {Z(r.fL3, 2)} & {r.reps} \\\\\n")
            f.write("\\addlinespace\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    e5.to_csv(os.path.join(PROC, "counters_hugepages.csv"), index=False)
# E4 kernels: bytes of cache-line traffic per element from the counters vs the compulsory 12 B
e4 = df[df.exp.str.startswith("E4_")].copy()
e4["kernel"] = e4.exp.str[3:]; e4["n"] = e4.label.str.replace("n=", "").astype(int)
e4["fill_bytes"] = 64 * (e4.fL2 + e4.fL3 + e4.fD)
e4.to_csv(os.path.join(PROC, "counters_kernels.csv"), index=False)
with open(os.path.join(GEN, "table_counters_kernels.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrrrr}\n\\toprule\nkernel & footprint (3 arrays) & ns/elem & cycles/elem & fills L2 / L3 / DRAM per elem & line-fill traffic (B/elem) & DTLB miss/elem & instr./elem \\\\\n\\midrule\n")
    for r in e4.sort_values(["kernel", "n"]).itertuples():
        f.write(f"{r.kernel} & {3 * r.n * 4 / 1024:.0f} KiB & {r.ns:.4f} & {r.cyc:.3f} & {Z4(r.fL2)} / {Z4(r.fL3)} / {Z4(r.fD)} & {max(r.fill_bytes, 0):.2f} & {Z4(max(r.dtlb, 0))} & {r.instr:.3f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
for col, lab, c in (("fL2", "L1D fills from L2", C["blue"]), ("fL3", "from L3", C["orange"]), ("fD", "from DRAM", C["red"]), ("dtlb", "DTLB misses", C["green"])):
    ax.plot(e1.kib, e1[col], "o-", ms=3.5, color=c, label=lab)
ax.set_xscale("log", base=2); ax.set_xlabel("cache footprint of the chase (KiB; compact 4 KiB pages)"); ax.set_ylabel("events per dependent load (counters)"); ax.set_ylim(-0.05, 1.3)
ax2 = ax.twinx(); ax2.plot(e1.kib, e1.ns, "k--s", ms=3, lw=1.0, label="measured ns / hop"); ax2.set_yscale("log"); ax2.set_ylabel("latency (ns/hop, log)"); ax2.grid(False)
for kb, nm in ((L1, "L1D"), (L2, "L2"), (L3, "L3")): ax.axvline(kb, color="gray", ls=":", lw=0.9); ax.text(kb * 1.05, 1.22, nm, fontsize=7.5, color="gray")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="center left")
ax.set_title("Where each dependent load is served (hardware counters) and what it costs", fontsize=9.5)
ax = axes[0, 1]
ax.plot(fit.pred, fit.ns, "o", ms=4, color=C["purple"]); lim = [0.2, 150]; ax.plot(lim, lim, "k--", lw=1)
ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.2, 150); ax.set_ylim(0.2, 150)
ax.set_xlabel("latency predicted from counters (ns)"); ax.set_ylabel("measured latency (ns/hop)")
ax.set_title("AMAT from counters: t = %.2f + %.1f L2 + %.1f L3 + %.1f DRAM + %.1f DTLB (ns); median error %.0f%%, max %.0f%%" % (coef[0], coef[1], coef[2], coef[3], coef[4], MEDERR, MAXERR), fontsize=8)
ax = axes[1, 0]
if len(e2):
    x = np.arange(len(e2)); ax.bar(x, e2.ns, color=[C["red"], C["orange"], C["green"]][:len(e2)]); ax.set_yscale("log"); ax.set_ylabel("ns per line (log)")
    for i, r in enumerate(e2.itertuples()): ax.text(i, r.ns * 1.15, f"{r.ns:.1f} ns\n{r.fD:.2f} DRAM fills/line\nMLP = {dep_ns / r.ns:.0f}", ha="center", fontsize=7.5)
    ax.set_xticks(x); ax.set_xticklabels([l.replace(" ", "\n", 1) for l in e2.label]); ax.set_ylim(top=e2.ns.max() * 8)
if len(e2): ax.set_title("Same miss count, different cost: %.0f vs %.1f vs %.1f ns per line (dependent / independent / sequential, 1 GiB)" % tuple(e2.ns.iloc[:3]), fontsize=9)
ax = axes[1, 1]
for (kib, g), c in zip(e3.groupby("kib"), (C["blue"], C["orange"], C["red"])):
    g = g.sort_values("pages"); ax.plot(g.pages, g.dtlb, "o-", color=c, label=f"{kib} KiB: DTLB miss / hop")
ax.set_xscale("log", base=2); ax.set_xlabel("4 KiB pages holding the same cache lines"); ax.set_ylabel("DTLB misses per dependent load")
ax2 = ax.twinx()
for (kib, g), c in zip(e3.groupby("kib"), (C["blue"], C["orange"], C["red"])):
    g = g.sort_values("pages"); ax2.plot(g.pages, g.ns, "s--", ms=3, color=c, alpha=0.7, label=f"{kib} KiB: ns/hop")
ax2.set_yscale("log"); ax2.set_ylabel("latency (ns/hop)"); ax2.grid(False)
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, fontsize=6.5, ncol=2, loc="upper left")
ax.set_title("TLB: misses per load (counters) and latency at constant cache footprint", fontsize=9.5)
fig.suptitle("Project 2 -- Fig. E: hardware-counter evidence (cache and TLB misses vs measured latency)", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figE_counters.png"), bbox_inches="tight"); plt.close(fig)
print("done; latency model: median error %.1f%%, max %.1f%%, log-R2 %.4f, coefficients (ns): %s" % (MEDERR, MAXERR, r2, dict(zip(names, np.round(coef, 2)))))
