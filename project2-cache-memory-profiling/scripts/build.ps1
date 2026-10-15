# Builds every Project 2 experiment binary with MinGW-w64 GCC.
$ErrorActionPreference = "Stop"
$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
$GXX = "C:\msys64\mingw64\bin\g++.exe"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
$bin = Join-Path $root "bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null

# -march=native: AVX-512 intrinsics (kern.hpp); no -ffast-math anywhere.
$flags = @("-O3", "-march=native", "-std=c++17", "-static", "-pthread")
Remove-Item (Join-Path $root "results\build_flags.txt") -ErrorAction SilentlyContinue
foreach ($n in @("exp1_latency_zeroqueue", "exp2_pattern_granularity", "exp3_rw_mix", "exp4_intensity_sweep",
                 "exp5_workingset_sweep", "exp6_cachemiss_impact", "exp7_tlbmiss_impact", "exp8_pitfall_demo", "exp9_counter_target", "freq_calibrate")) {
    Write-Output "building $n ..."
    & $GXX @flags -I $src -o (Join-Path $bin "$n.exe") (Join-Path $src "$n.cpp")
    if ($LASTEXITCODE -ne 0) { throw "build failed: $n" }
    "$n : $GXX $($flags -join ' ')" | Add-Content (Join-Path $root "results\build_flags.txt")
}
& $GXX --version | Select-Object -First 1 | Out-File -Encoding utf8 (Join-Path $root "results\toolchain_versions.txt")
Write-Output "build complete."
