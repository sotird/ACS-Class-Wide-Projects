#!/usr/bin/env python3
"""Hardware-counter evidence for Project 1 (exp9): raw counter CSV -> figE_counters.png, processed CSV, LaTeX tables.

Input : data/raw/counters/exp9_counters.csv  (scripts/run_counters.ps1; counts per ELEMENT, setup differenced out)
Output: results/figE_counters.png, data/processed/counters_*.csv, report/generated/table_counters_*.tex

What the counters add to the compiler reports / disassembly: they MEASURE (1) retired instructions per element (a vectorized loop retires
~1/lanes as many instructions), (2) cycles per element and IPC, (3) loads per element, (4) cache-line traffic from each level
(L1D fills from L2 / L3 / DRAM; x64 B = bytes moved), which can be checked against the roofline's traffic model (compulsory vs write-allocate).
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
C = {"scalar": "#777777", "sse2": "#8172b3", "avx2": "#55a868", "avx512": "#c44e52", "avx512_fm": "#dd8452"}
df = pd.read_csv(os.path.join(RAW, "exp9_counters.csv"))
df = df.rename(columns={"TotalCycles_per_elem": "cyc", "InstructionRetired_per_elem": "instr", "DcacheAccesses_per_elem": "loads", "FillL2_per_elem": "fL2",
                        "FillL3_per_elem": "fL3", "FillDram_per_elem": "fD", "ns_per_elem": "ns"})
df["ipc"] = df.instr / df.cyc
df["fill_bytes"] = 64 * (df.fL2 + df.fL3 + df.fD)
df["level"] = df.n.map({2048: "L1", 65536: "L2", 1048576: "L3", 33554432: "DRAM"})
df.to_csv(os.path.join(PROC, "counters_all.csv"), index=False)
KER = ["saxpy", "mul", "stencil", "dot"]
VAR = ["scalar", "sse2", "avx2", "avx512"]
LANES = {"scalar": 1, "sse2": 4, "avx2": 8, "avx512": 16, "avx512_fm": 16}
# compulsory bytes per element (read + write, STREAM convention) and bytes of line traffic INCLUDING write-allocate reads (reads only: fills count reads)
READ_FILL_B = {"saxpy": 8, "mul": 12, "stencil": 8, "dot": 8}    # x,y ; x,y + RFO of z ; x + RFO of y ; x,y
def g(variant, kernel, level, col):
    r = df[(df.variant == variant) & (df.kernel == kernel) & (df.level == level)]
    return float(r[col].iloc[0]) if len(r) else np.nan

# ------------------------------------------------------------------ table 1: instructions per element in L1 (SIMD actually happened)
with open(os.path.join(GEN, "table_counters_instr.tex"), "w") as f:
    f.write("\\begin{tabular}{llrrrrr}\n\\toprule\nkernel (f32, L1) & build & instr./elem & reduction vs.\\ scalar & cycles/elem & IPC & loads/elem \\\\\n\\midrule\n")
    for k in KER:
        base = g("scalar", k, "L1", "instr")
        for v in VAR + (["avx512_fm"] if k == "dot" else []):
            i = g(v, k, "L1", "instr")
            if np.isnan(i): continue
            f.write(f"{k if v == 'scalar' else ''} & {v.replace('_', '-')} & {i:.3f} & {base / i:.1f}$\\times$ & {g(v, k, 'L1', 'cyc'):.3f} & {g(v, k, 'L1', 'ipc'):.2f} & {g(v, k, 'L1', 'loads'):.3f} \\\\\n")
        f.write("\\addlinespace\n")
    f.write("\\bottomrule\n\\end{tabular}\n")
# ------------------------------------------------------------------ table 2: line traffic at DRAM size vs model
with open(os.path.join(GEN, "table_counters_traffic.tex"), "w") as f:
    f.write("\\begin{tabular}{lrrrrr}\n\\toprule\nkernel (f32, 384 MiB) & model: line reads (B/elem) & measured scalar & measured AVX-512 & DRAM share of fills & cycles/elem scalar $\\rightarrow$ AVX-512 \\\\\n\\midrule\n")
    for k in KER:
        s, a = g("scalar", k, "DRAM", "fill_bytes"), g("avx512", k, "DRAM", "fill_bytes")
        share = g("avx512", k, "DRAM", "fD") / max(1e-12, g("avx512", k, "DRAM", "fL2") + g("avx512", k, "DRAM", "fL3") + g("avx512", k, "DRAM", "fD"))
        f.write(f"{k} & {READ_FILL_B[k]} & {s:.2f} & {a:.2f} & {100 * share:.0f}\\% & {g('scalar', k, 'DRAM', 'cyc'):.2f} $\\rightarrow$ {g('avx512', k, 'DRAM', 'cyc'):.2f} \\\\\n")
    f.write("\\bottomrule\n\\end{tabular}\n")

# ------------------------------------------------------------------ figure
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
ax = axes[0, 0]
w = 0.2
for j, v in enumerate(VAR):
    vals = [g(v, k, "L1", "instr") for k in KER]
    ax.bar(np.arange(len(KER)) + (j - 1.5) * w, vals, w, color=C[v], label=f"{v} ({LANES[v]} lanes)")
    for i, x in enumerate(vals):
        if not np.isnan(x): ax.text(i + (j - 1.5) * w, x * 1.04, f"{x:.2f}", ha="center", fontsize=6.5)
ax.set_xticks(range(len(KER))); ax.set_xticklabels(KER); ax.set_yscale("log"); ax.set_ylabel("retired instructions per element (log)")
ax.set_title("SIMD measured, not assumed: retired instructions per element (L1-resident, f32)", fontsize=9.5); ax.legend(fontsize=7)
ax = axes[0, 1]
levels = ["L1", "L2", "L3", "DRAM"]
for k, ls in zip(("saxpy", "mul"), ("-", "--")):
    for v in ("scalar", "avx512"):
        ax.plot(range(4), [g(v, k, l, "cyc") for l in levels], ls, marker="o", ms=4, color=C[v], label=f"{k} {v}")
ax.set_xticks(range(4)); ax.set_xticklabels(levels); ax.set_yscale("log"); ax.set_ylabel("cycles per element (log)"); ax.legend(fontsize=7, ncol=2)
ax.set_title("Cycles per element by residency level (counter-based, no clock assumption)", fontsize=9.5)
ax = axes[1, 0]
x = np.arange(len(KER))
for j, v in enumerate(("scalar", "avx512")):
    ax.bar(x + (j - 0.5) * 0.3, [g(v, k, "DRAM", "fill_bytes") for k in KER], 0.3, color=C[v], label=f"measured line fills, {v}")
ax.plot(x, [READ_FILL_B[k] for k in KER], "k_", ms=22, mew=2, label="model: compulsory reads + write-allocate")
ax.set_xticks(x); ax.set_xticklabels(KER); ax.set_ylabel("bytes of cache-line fills per element (DRAM-resident)"); ax.legend(fontsize=7)
ax.set_title("Memory traffic measured by the counters vs the roofline's traffic model", fontsize=9.5)
ax = axes[1, 1]
for v in VAR + ["avx512_fm"]:
    vals = [g(v, "dot", l, "cyc") for l in levels]
    if not np.all(np.isnan(vals)): ax.plot(range(4), vals, "o-", ms=4, color=C[v], label=f"dot {v.replace('_', '-')}")
ax.set_xticks(range(4)); ax.set_xticklabels(levels); ax.set_yscale("log"); ax.set_ylabel("cycles per element (log)"); ax.legend(fontsize=7)
ax.set_title("Dot product: 'vectorized' builds stay at ~2 cycles/element (dependent FP adds) until -ffast-math", fontsize=9)
fig.suptitle("Project 1 -- Fig. E: hardware-counter evidence (instructions, cycles, cache-line traffic)", fontweight="bold", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(os.path.join(RES, "figE_counters.png"), bbox_inches="tight"); plt.close(fig)
print("done")
