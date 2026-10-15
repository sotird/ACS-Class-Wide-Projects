# Project 3 end-to-end (run from an ELEVATED PowerShell: the SMART temperature counters need administrator rights):
# build the C++ tool, record the environment, run all FIO stages (resume-safe), the C++ reference runs,
# then sanity checks + processing + plots.  ~4.5 h on the test machine.
#   powershell -File scripts\run_all.ps1            (add -SkipFio to only re-process)
param([switch]$SkipFio)
$ErrorActionPreference = "Stop"
$py = "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
$root = Split-Path -Parent $PSScriptRoot
& (Join-Path $PSScriptRoot "build.ps1")
& (Join-Path $PSScriptRoot "capture_env.ps1") | Out-Null
if (-not $SkipFio) {
    & $py (Join-Path $PSScriptRoot "run_fio.py") --stages 1
    & (Join-Path $PSScriptRoot "run_ref.ps1")
    & $py (Join-Path $PSScriptRoot "run_fio.py") --stages 2,3,4,5,6,7,8,9,10,12,11
}
& $py (Join-Path $PSScriptRoot "sanity_check.py")
& $py (Join-Path $PSScriptRoot "process_and_plot.py")
