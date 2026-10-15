# Project 2: Cache & Memory Performance Profiling

Measures the memory hierarchy in three ways. Dependent pointer chases obtain zero-queue latency, AVX-512 streaming kernels show bandwidth, and an MLC-style loaded-latency method runs pinned generator threads alongside a latency probe. Hardware counters then tie the results to cache and TLB behaviour. The method, environment, results and the commands to reproduce them are in ``report/report.pdf``.

## Layout

### `src/`

One program per experiment plus shared headers.

| file | content |
|---|---|
| `common.hpp` | timing, pinning, allocation on 4 KiB and 2 MiB pages, statistics, CSV output |
| `chase.hpp` | pointer-chase layouts and the dependent-load loop |
| `kern.hpp` | AVX-512 streaming kernels |
| `mt.hpp` | multi-threaded load generator and latency probe |
| `exp1_latency_zeroqueue.cpp` | latency vs. footprint, L1 to DRAM |
| `exp2_pattern_granularity.cpp` | sequential/strided/random access and block-size sweep |
| `exp3_rw_mix.cpp` | read/write mixes, throughput and latency |
| `exp4_intensity_sweep.cpp` | loaded-latency sweep (throughput-latency curve) |
| `exp5_workingset_sweep.cpp` | single-core bandwidth vs. working set |
| `exp6_cachemiss_impact.cpp` | cache-miss impact from runtime |
| `exp7_tlbmiss_impact.cpp` | TLB-miss impact with 4 KiB and 2 MiB pages, and a working-set sweep on both |
| `exp8_pitfall_demo.cpp` | benchmark pitfalls demonstrated with data |
| `exp9_counter_target.cpp` | target program for the hardware-counter runs |
| `freq_calibrate.cpp` | measures the sustained core clock |

### `scripts/`

| script | role |
|---|---|
| `build.ps1` | builds every binary into `bin/` (flags are written to `results/build_flags.txt`) |
| `capture_env.ps1` | records the measurement environment to `results/environment.txt` |
| `run_all.ps1` | runs all experiments into `data/raw` (`-RawDir`/`-Subset` for the repeat runs) |
| `process_and_plot.py` | raw data -> `data/processed`, `results/fig*.png`, `report/generated/*.tex` |
| `repeat_check.py` | compares the independent runs |
| `set_balance_check.py` | offline cache-set balance of the exp7 layouts on 2 MiB pages |
| `run_counters.ps1` | hardware-counter runs (elevated PowerShell; uses `../tools`; `-Only <group>` re-runs one group) |
| `process_counters.py` | counter data -> `results/figE_counters.png`, `report/generated/table_counters_*.tex` |

### `data/`

| path | content |
|---|---|
| `raw/` | one CSV per experiment, `cache_topology*.csv` |
| `raw/exp7_tlb.csv`, `raw/exp7_hugepage_wsweep.csv` | exp7 results (4 KiB and 2 MiB pages) |
| `raw/exp7_tlb_unpermuted_layout.csv` | exp7 with the first, address-order page layout (kept as a pitfall exhibit) |
| `raw/counters/` | hardware-counter runs |
| `repeats/run2/`, `repeats/run3/` | headline experiments from two further independent runs |
| `processed/` | summary tables derived from the raw data |

### `results/`

| path | content |
|---|---|
| `figA..figD_*.png` | report figures A-D: latency and working set; pattern, granularity and read/write mix; loaded-latency curve; cache-miss, TLB-miss and pitfalls |
| `figE_counters.png` | hardware-counter figure (the same data as the counter tables; not included in the report) |
| `environment.txt`, `power_plan.txt`, `frequency_log.csv`, `frequency_by_cpu.csv`, `toolchain_versions.txt`, `build_flags.txt` | recorded environment and exact compiler flags |

### `report/`

| path | content |
|---|---|
| `report.tex`, `report.pdf` | the report |
| `generated/` | LaTeX tables written by the processing scripts |

`../tools/` holds the counter-recording tools used by `run_counters.ps1`.
