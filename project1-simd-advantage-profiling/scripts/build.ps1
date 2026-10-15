# Builds every Project 1 experiment binary with MinGW-w64 GCC.
# Every ISA variant of an experiment is compiled from the IDENTICAL source
# file; only the compiler flags (and the -DBUILD_TAG label) differ, so any
# measured difference is attributable to vectorization / ISA, not to
# different hand-written code.
$ErrorActionPreference = "Stop"
$env:Path = "C:\msys64\mingw64\bin;" + $env:Path   # cc1plus needs its sibling DLLs on PATH
$GXX = "C:\msys64\mingw64\bin\g++.exe"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
$bin = Join-Path $root "bin"
New-Item -ItemType Directory -Force -Path $bin | Out-Null

$common = @("-std=c++17", "-static")
$variants = @{
    # -march=native on the scalar baseline keeps scalar FMA etc. available, so
    # the scalar-vs-vector gap measures SIMD only, not "FMA vs no FMA".
    "scalar"    = @("-O3", "-march=native", "-fno-tree-vectorize", "-fno-tree-slp-vectorize")
    "sse2"      = @("-O3", "-march=x86-64")                      # gcc default target: SSE2, 128-bit
    "avx2"      = @("-O3", "-mavx2", "-mfma")                    # 256-bit
    "avx512"    = @("-O3", "-march=native")                      # widest available (znver5: 512-bit)
    "avx512_fm" = @("-O3", "-march=native", "-ffast-math")       # + FP reassociation allowed
    "avx512_gather" = @("-O3", "-march=native", "-mtune-ctrl=use_gather")   # tuning override: allow vgatherdps (gcc declines gathers on znver5 by default)
}

function Build([string]$exp, [string[]]$sources, [string]$variant) {
    $srcPaths = @($sources | ForEach-Object { Join-Path $src $_ })
    $outName = if ($variant) { "${exp}_${variant}" } else { $exp }
    $flags = if ($variant) { $variants[$variant] } else { $variants["avx512"] }
    $tag = if ($variant) { $variant } else { "avx512" }
    $out = Join-Path $bin "$outName.exe"
    Write-Output "building $outName ..."
    & $GXX @flags @common "-DBUILD_TAG=\`"$tag\`"" -I $src -o $out @srcPaths
    if ($LASTEXITCODE -ne 0) { throw "build failed: $outName" }
    "$outName : $GXX $($flags -join ' ') $($common -join ' ')" | Add-Content (Join-Path $root "results\build_flags.txt")
}

New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null
Remove-Item (Join-Path $root "results\build_flags.txt") -ErrorAction SilentlyContinue

foreach ($v in @("scalar", "sse2", "avx2", "avx512", "avx512_fm")) { Build "exp1" @("exp1_scalar_vs_vector.cpp") $v }
foreach ($v in @("scalar", "avx2", "avx512"))                      { Build "exp2" @("exp2_locality_sweep.cpp") $v }
foreach ($v in @("scalar", "sse2", "avx2", "avx512"))              { Build "exp3" @("exp3_alignment_tail.cpp") $v }
Build "exp3b" @("exp3b_tail_rotation.cpp") "avx512"      # follow-up: separates the masked tail from the store-to-load dependency between calls
foreach ($v in @("scalar", "avx512", "avx512_gather"))             { Build "exp4" @("exp4_stride_gather.cpp") $v }
foreach ($v in @("scalar", "avx512"))                              { Build "exp5" @("exp5_dtype_compare.cpp") $v }
foreach ($v in @("scalar", "sse2", "avx2", "avx512", "avx512_fm"))  { Build "exp9_counters" @("exp9_counters.cpp") $v }
Build "exp7a" @("exp7a_bandwidth_roofline.cpp") "avx512"
Build "exp7b" @("exp7b_flops_roofline.cpp") "avx512"
Build "exp8"  @("exp8_negative_novec.cpp", "black_box.cpp") "avx512"
Build "freq_calibrate" @("freq_calibrate.cpp") ""

& $GXX --version | Select-Object -First 1 | Out-File -Encoding utf8 (Join-Path $root "results\toolchain_versions.txt")
Write-Output "build complete."
