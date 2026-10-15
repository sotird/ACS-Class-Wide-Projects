# Runs every Project 1 experiment binary (all ISA variants) sequentially and
# writes raw CSVs to data\raw\. Records the environment and the measured core
# clock before and after. Never modifies system settings.
# Run on an otherwise idle machine; experiments are pinned (see common.hpp).
param([string]$RawDir = "data\raw")   # repeat runs: -RawDir data\repeats\run2
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "bin"
$raw = Join-Path $root $RawDir
$primary = ($RawDir -eq "data\raw")
$results = Join-Path $root "results"
New-Item -ItemType Directory -Force -Path $raw | Out-Null
New-Item -ItemType Directory -Force -Path $results | Out-Null

if ($primary) { & (Join-Path $PSScriptRoot "capture_env.ps1") | Out-Null }
$freqLog = if ($primary) { Join-Path $results "frequency_log.csv" } else { Join-Path $raw "frequency_log.csv" }
"phase,ghz" | Out-File -Encoding ascii $freqLog
function LogFreq($phase) { (& (Join-Path $bin "freq_calibrate.exe") 3) | ForEach-Object { "$phase,$_" | Add-Content $freqLog } }
LogFreq "start"

function Run($exe, [string[]]$exeArgs) {
    Write-Output "running $exe ..."
    & (Join-Path $bin "$exe.exe") @exeArgs
    if ($LASTEXITCODE -ne 0) { throw "run failed: $exe (exit $LASTEXITCODE)" }
}

foreach ($v in @("scalar", "sse2", "avx2", "avx512", "avx512_fm")) { Run "exp1_$v" @((Join-Path $raw "exp1_$v.csv")) }
LogFreq "after_exp1"
foreach ($v in @("scalar", "avx2", "avx512")) {
    Run "exp2_$v" @((Join-Path $raw "exp2_$v.csv"), (Join-Path $raw "cache_topology.csv"))
}
foreach ($v in @("scalar", "sse2", "avx2", "avx512")) { Run "exp3_$v" @((Join-Path $raw "exp3_$v.csv")) }
Run "exp3b_avx512" @((Join-Path $raw "exp3b_tail_rotation.csv"))
foreach ($v in @("scalar", "avx512", "avx512_gather")) { Run "exp4_$v" @((Join-Path $raw "exp4_$v.csv")) }
foreach ($v in @("scalar", "avx512"))                 { Run "exp5_$v" @((Join-Path $raw "exp5_$v.csv")) }
Run "exp7a_avx512" @((Join-Path $raw "exp7a_bandwidth.csv"))
Run "exp7b_avx512" @((Join-Path $raw "exp7b_peak_flops.csv"))
Run "exp8_avx512"  @((Join-Path $raw "exp8_negative.csv"))
LogFreq "end"
Write-Output "all experiments complete. raw data in $raw"
