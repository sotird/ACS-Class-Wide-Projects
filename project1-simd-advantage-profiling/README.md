# Project 1: SIMD Advantage Profiling

Compares scalar and auto-vectorized builds (SSE2, AVX2, AVX-512) of SAXPY, dot product, elementwise multiply and a three-point
stencil. The only thing we change here is the compiler flags. Method, environment, results and the
commands to reproduce them are found in the report under ``report/report.pdf``.

## Layout

### `src/`

One program per experiment (built once per ISA variant) plus shared headers.

| file | content |
|---|---|
| `kernels.hpp` | kernels for the benchmarks |
| `common.hpp` | timing, pinning, aligned allocation, CSV output |
| `validate.hpp` | reference implementations |
| `exp1_scalar_vs_vector.cpp` | scalar vs. vector builds across working-set sizes, all kernels |
| `exp2_locality_sweep.cpp` | fine working-set sweep for SAXPY |
| `exp3_alignment_tail.cpp` | alignment offsets and short-vector call times |
| `exp3b_tail_rotation.cpp` | follow-up to exp3 with rotating output buffers |
| `exp4_stride_gather.cpp` | strided and gather access |
| `exp5_dtype_compare.cpp` | f32, f64 and i32 |
| `exp6_vecreport.cpp` | compiled only for compiler vectorization reports and disassembly |
| `exp7a_bandwidth_roofline.cpp` | measured bandwidth ceilings per cache level |
| `exp7b_flops_roofline.cpp` | measured FMA peak per vector width |
| `exp8_negative_novec.cpp` | loop hidden behind an opaque call that shows a failure case, needs `black_box.cpp` |
| `exp9_counters.cpp` | target program for the hardware-counter runs |
| `freq_calibrate.cpp` | measures the sustained core clock |

### `scripts/`

| script | role |
|---|---|
| `build.ps1` | builds every binary into `bin/`, flags are written to `results/build_flags.txt` |
| `capture_env.ps1` | records the measurement environment to `results/environment.txt` |
| `run_all.ps1` | runs all experiments into `data/raw` unless a `-RawDir` was specified. |
| `verify_vectorization.ps1` | compiler reports and disassembly excerpts into `results/vectorization_evidence` |
| `process_and_plot.py` | raw data -> `data/processed`, `results/fig*.png`, `report/generated/*.tex` |
| `repeat_check.py` | compares the independent full runs |
| `run_counters.ps1` | hardware-counter runs (elevated PowerShell; uses `../tools`) |
| `process_counters.py` | counter data -> `results/figE_counters.png`, `report/generated/table_counters_*.tex` |

### `data/`

| path | content |
|---|---|
| `raw/` | one CSV per experiment and ISA variant (`exp<N>_<variant>.csv`), `cache_topology.csv`, `counters/` |
| `repeats/run2/`, `repeats/run3/` | the same files from two further independent runs of the suite |
| `processed/` | summary tables derived from the raw data |

### `results/`

| path | content |
|---|---|
| `figA..figD_*.png` | report figures A-D: scalar vs. vector and locality; alignment and tail; stride/gather and data type; roofline and negative example |
| `figE_counters.png` | hardware-counter figure (the same data as the counter tables; not included in the report) |
| `environment.txt`, `power_plan.txt`, `frequency_log.csv`, `toolchain_versions.txt`, `build_flags.txt` | recorded environment and exact compiler flags |
| `vectorization_evidence/` | compiler vectorization reports and disassembly excerpts |

### `report/`

| path | content |
|---|---|
| `report.tex`, `report.pdf` | the report |
| `generated/` | LaTeX tables written by the processing scripts |

`../tools/` holds the counter-recording tools used by `run_counters.ps1`.
