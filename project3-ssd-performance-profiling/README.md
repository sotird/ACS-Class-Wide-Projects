# Project 3: SSD Performance Profiling

Profiles an SSD with FIO, using the Windows windowsaio engine with direct I/O, and checks it against an independent C++/Win32 reference tool. The measurements cover QD=1 latency, block-size and pattern sweeps, read/write mixes, queue-depth sweeps, tail latency, LBA range, device state and misleading setups. The method, environment, results and the commands to reproduce them are in ``report/report.pdf``.

## Layout

### `src/`

| file | content |
|---|---|
| `ref_qd1.cpp` | independent QD=1 reference tool and misleading-setup demonstrations (Win32, no FIO) |

### `scripts/`

| script | role |
|---|---|
| `run_fio.py` | orchestrates all FIO stages 1-12 on the scratch file; safety checks, SMART polling, resume-safe |
| `run_all.ps1` | end to end: build, record environment, FIO stages, C++ reference runs, plots |
| `build.ps1` | builds the C++ tool into `bin/` |
| `capture_env.ps1` | records the measurement environment to `results/environment.txt` |
| `run_ref.ps1` | runs the C++ reference tool |
| `monitor_os.ps1` | 1-second OS disk and CPU counters while a benchmark runs |
| `rerun_perturbed.py` | deletes and re-runs the jobs that overlapped other work on the host |
| `sanity_check.py` | parser self-test and consistency checks on the raw data |
| `process_and_plot.py` | raw JSON/CSV -> `data/processed`, `results/fig*.png`, `report/generated/*.tex` |

### `data/`

| path | content |
|---|---|
| `raw/jobs/` | the FIO job file of every run |
| `raw/json/` | raw FIO JSON output of every run |
| `raw/logs/` | per-second FIO logs and `smart_log.csv` (SMART temperature log) |
| `raw/ref/` | CSVs of the C++ reference tool |
| `raw/job_times.csv` | wall-clock start and end of every job |
| `raw/perturbation_windows.txt`, `raw/perturbed_jobs_rerun.txt` | jobs that were discarded and re-run |
| `raw/*.done` | markers of completed preconditioning and idle rests |
| `processed/` | parsed and summarized tables derived from the raw data |
| `archive_80MiB_nonadmin_campaign/` | the earlier pilot campaign (own scripts, data, results and report) |

### `results/`

| path | content |
|---|---|
| `figA..figE_*.png` | report figures A-E: baselines and block size; queue-depth sweep and mix; tail latency; device state; read stability and misleading setups |
| `environment.txt` | recorded environment |

### `report/`

| path | content |
|---|---|
| `report.tex`, `report.pdf` | the report |
| `generated/` | LaTeX tables written by the processing scripts |
