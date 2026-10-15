# Runs every Project 2 experiment sequentially -> data\raw\*.csv. Records the
# environment and measured core clock before/after. Never modifies system
# settings. Run on an otherwise idle machine (the multi-threaded experiments
# use the even logical CPUs 0-30, one per physical core; experiments are pinned, see src\common.hpp / mt.hpp).
#   -RawDir data\repeats\run2 -Subset   : repeat only the headline experiments
param([string]$RawDir = "data\raw", [switch]$Subset)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "bin"
$raw = Join-Path $root $RawDir
$results = Join-Path $root "results"
$primary = ($RawDir -eq "data\raw")
New-Item -ItemType Directory -Force -Path $raw | Out-Null
New-Item -ItemType Directory -Force -Path $results | Out-Null
if ($primary) { & (Join-Path $PSScriptRoot "capture_env.ps1") | Out-Null }
$freqLog = if ($primary) { Join-Path $results "frequency_log.csv" } else { Join-Path $raw "frequency_log.csv" }
"phase,ghz" | Out-File -Encoding ascii $freqLog
function LogFreq($phase) { (& (Join-Path $bin "freq_calibrate.exe") 3 4) | ForEach-Object { "$phase,$_" | Add-Content $freqLog } }
function Run($exe, [string[]]$exeArgs) {
    Write-Output "running $exe $($exeArgs | ForEach-Object { Split-Path $_ -Leaf }) ..."
    & (Join-Path $bin "$exe.exe") @exeArgs
    if ($LASTEXITCODE -ne 0) { throw "run failed: $exe (exit $LASTEXITCODE)" }
}
LogFreq "start"
Run "exp1_latency_zeroqueue" @((Join-Path $raw "exp1_latency_cpu4.csv"), (Join-Path $raw "cache_topology.csv"), "4")
if (-not $Subset) { Run "exp1_latency_zeroqueue" @((Join-Path $raw "exp1_latency_cpu20.csv"), (Join-Path $raw "cache_topology_cpu20.csv"), "20") }
LogFreq "after_exp1"
if (-not $Subset) { Run "exp2_pattern_granularity" @((Join-Path $raw "exp2_latency.csv"), (Join-Path $raw "exp2_bandwidth.csv"), "4") }
Run "exp3_rw_mix" @((Join-Path $raw "exp3_rw_mix.csv"))
Run "exp4_intensity_sweep" @((Join-Path $raw "exp4_intensity.csv"))
Run "exp5_workingset_sweep" @((Join-Path $raw "exp5_workingset.csv"), "4")
Run "exp6_cachemiss_impact" @((Join-Path $raw "exp6_cachemiss.csv"), "4")
if (-not $Subset) {
    # 4 KiB and 2 MiB pages (the huge-page arm needs SeLockMemoryPrivilege; without it only the 4 KiB arm runs); 2nd CSV = working-set sweep
    Run "exp7_tlbmiss_impact" @((Join-Path $raw "exp7_tlb.csv"), "4", (Join-Path $raw "exp7_hugepage_wsweep.csv"))
    Run "exp8_pitfall_demo" @((Join-Path $raw "exp8_cache_reuse.csv"), (Join-Path $raw "exp8_pinning.csv"), (Join-Path $raw "exp8_first_touch.csv"), (Join-Path $raw "exp8_address_generation.csv"))
}
LogFreq "end"
Write-Output "all experiments complete. raw data in $raw"
