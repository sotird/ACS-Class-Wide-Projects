# Generates CONCISE vectorization evidence (compiler reports + targeted
# disassembly excerpts) for every kernel and ISA variant, plus the negative
# example. Writes short curated files to results\vectorization_evidence\ --
# never full dumps. Run when the machine is otherwise idle (compiles only).
$ErrorActionPreference = "Stop"
$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
$GXX = "C:\msys64\mingw64\bin\g++.exe"
$OBJDUMP = "C:\msys64\mingw64\bin\objdump.exe"
$root = Split-Path -Parent $PSScriptRoot
$src = Join-Path $root "src"
$out = Join-Path $root "results\vectorization_evidence"
Remove-Item -Recurse -Force $out -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path $out | Out-Null

$variants = [ordered]@{
    "scalar"    = @("-O3", "-march=native", "-fno-tree-vectorize", "-fno-tree-slp-vectorize")
    "sse2"      = @("-O3", "-march=x86-64")
    "avx2"      = @("-O3", "-mavx2", "-mfma")
    "avx512"    = @("-O3", "-march=native")
    "avx512_fm" = @("-O3", "-march=native", "-ffast-math")
    "avx512_gather" = @("-O3", "-march=native", "-mtune-ctrl=use_gather")
}

# map kernels.hpp line numbers -> kernel names (nearest preceding 'inline ... name(' line)
$kLines = Get-Content (Join-Path $src "kernels.hpp")
$lineToKernel = @{}
$cur = ""
for ($i = 0; $i -lt $kLines.Count; $i++) {
    if ($kLines[$i] -match 'inline\s+\S+\s+(\w+)\(') { $cur = $Matches[1] }
    elseif ($kLines[$i] -match 'inline\s+void\s+(\w+)\(') { $cur = $Matches[1] }
    if ($kLines[$i] -match '^\s*for \(') { $lineToKernel[$i + 1] = $cur }
}

$summary = New-Object System.Collections.Generic.List[string]
$summary.Add("| variant | kernel | main loop vector width (bytes) | epilogue handling |")
$summary.Add("|---|---|---|---|")
foreach ($v in $variants.Keys) {
    $flags = $variants[$v]
    $exe = Join-Path $out "exp6_$v.exe"
    $opt = Join-Path $out "vec_optimized_$v.txt"
    $mis = Join-Path $out "vec_missed_$v.txt"
    & $GXX @flags -std=c++17 -I $src "-fopt-info-vec-optimized=$opt" -o $exe (Join-Path $src "exp6_vecreport.cpp")
    if ($LASTEXITCODE -ne 0) { throw "report build failed: $v" }
    & $GXX @flags -std=c++17 -I $src "-fopt-info-vec-missed=$mis" -o "$out\_tmp.exe" (Join-Path $src "exp6_vecreport.cpp")
    Remove-Item "$out\_tmp.exe" -ErrorAction SilentlyContinue

    $perKernel = @{}
    if (Test-Path $opt) {
        Get-Content $opt | ForEach-Object {
            if ($_ -match 'kernels\.hpp:(\d+):\d+: optimized: (epilogue loop|loop) vectorized using (masked )?(\d+) byte vectors') {
                $k = $lineToKernel[[int]$Matches[1]]; if (-not $k) { return }
                if (-not $perKernel.ContainsKey($k)) { $perKernel[$k] = @{ main = New-Object System.Collections.Generic.SortedSet[int]; epi = New-Object System.Collections.Generic.SortedSet[string] } }
                if ($Matches[2] -eq 'loop') { [void]$perKernel[$k].main.Add([int]$Matches[4]) }
                else { [void]$perKernel[$k].epi.Add($(if ($Matches[3]) { "masked $($Matches[4])B" } else { "$($Matches[4])B" })) }
            }
        }
    }
    foreach ($k in @("saxpy", "dot", "mul", "stencil3", "mul_strided_c", "mul_gather")) {
        if ($perKernel.ContainsKey($k)) {
            $summary.Add("| $v | $k | $(($perKernel[$k].main | ForEach-Object { $_ }) -join '/') | $(($perKernel[$k].epi) -join ', ') |")
        } else {
            $summary.Add("| $v | $k | NOT VECTORIZED | - |")
        }
    }
}
$summary | Out-File -Encoding utf8 (Join-Path $out "summary_table.md")

function Excerpt([string]$exe, [string]$symRegex, [string]$mnemRegex, [string]$outFile, [string]$title, [int]$maxLines = 14) {
    $sym = & $OBJDUMP -t $exe | Select-String $symRegex | Where-Object { $_ -notmatch '\.(text|xdata|pdata)\$' } | ForEach-Object { ($_ -split '\s+')[-1] } | Select-Object -First 1
    if (-not $sym) { "(symbol $symRegex not found)" | Out-File $outFile; return }
    $dis = & $OBJDUMP -d --disassemble="$sym" -M intel $exe
    $hits = $dis | Select-String $mnemRegex | Select-Object -First $maxLines
    @("$title", "symbol: $sym", "(only lines matching /$mnemRegex/; full dump omitted)", "") + ($hits | ForEach-Object { $_.ToString().Trim() }) | Out-File -Encoding utf8 $outFile
}

$vec = "zmm|ymm|vfmadd|vmulps|vaddps|vmovups|vmovaps|vbroadcast"
Excerpt "$out\exp6_avx512.exe" 'saxpyIfEE' $vec "$out\saxpy_avx512_disasm.txt" "saxpy<float>, -O3 -march=native (AVX-512 FMA + masked epilogue)"
Excerpt "$out\exp6_avx512.exe" '_ZN7kernels3dotIfEET_PKS1_S3_y' 'vmulps|vaddss|vaddps|vfmadd|vextract|vshuf|vunpck|vmovshdup|vpermilps|zmm' "$out\dot_f32_default_disasm.txt" "dot<float>, default FP semantics (-O3 -march=native): vector multiplies, then SEQUENTIAL scalar adds" 16
Excerpt "$out\exp6_avx512_fm.exe" '_ZN7kernels3dotIfEET_PKS1_S3_y' 'vfmadd|vaddps|vmulps|zmm' "$out\dot_f32_fastmath_disasm.txt" "dot<float>, -ffast-math: independent vector accumulators (vfmadd231ps)" 12
Excerpt "$out\exp6_avx512.exe" '_ZN7kernels3dotIjEET_PKS1_S3_y' 'vpaddd|vpmulld|zmm' "$out\dot_i32_disasm.txt" "dot<uint32_t> (integer: associative, vectorizes with NO fast-math)" 10
Excerpt "$out\exp6_avx512_gather.exe" 'mul_gatherIfEE' 'vgather|vpgather|vmulps|zmm' "$out\gather_forced_disasm.txt" "mul_gather<float>, -mtune-ctrl=use_gather (gcc emits vgatherdps only when told to)" 10
Excerpt "$out\exp6_avx512.exe" 'mul_strided_cIfLy4EEE' 'vgather|vperm|vshuf|vpermt2ps|vpermi2ps|vmulps|zmm' "$out\stride4_disasm.txt" "mul_strided_c<float,4>" 10

# Negative example: opaque call (via_call) vs identical arithmetic inlined
$exp8 = Join-Path $out "exp8_report.exe"
$o8 = Join-Path $out "exp8_vec_optimized.txt"; $m8 = Join-Path $out "exp8_vec_missed.txt"
& $GXX -O3 -march=native -std=c++17 -I $src "-fopt-info-vec-optimized=$o8" -o $exp8 (Join-Path $src "exp8_negative_novec.cpp") (Join-Path $src "black_box.cpp")
& $GXX -O3 -march=native -std=c++17 -I $src "-fopt-info-vec-missed=$m8" -o "$out\_tmp8.exe" (Join-Path $src "exp8_negative_novec.cpp") (Join-Path $src "black_box.cpp")
Remove-Item "$out\_tmp8.exe" -ErrorAction SilentlyContinue
$lineCall = (Select-String -Path (Join-Path $src "exp8_negative_novec.cpp") -Pattern "y\[i\] = black_box").LineNumber
$lineInl  = (Select-String -Path (Join-Path $src "exp8_negative_novec.cpp") -Pattern "y\[i\] = x\[i\] \* 1\.0000001f").LineNumber
$missedCall = Get-Content $m8 | Select-String "exp8_negative_novec.cpp:${lineCall}:" | Select-Object -First 5
$optInl = Get-Content $o8 | Select-String "exp8_negative_novec.cpp:$($lineInl - 0):" | Select-Object -First 3
@("--- via_call() loop (source line $lineCall): calls black_box(), an opaque function in a separate TU ---") + ($missedCall | ForEach-Object { ($_ -replace '^.*exp8_negative_novec.cpp', 'exp8_negative_novec.cpp') }) +
@("", "--- inlined() loop (source line $lineInl): identical arithmetic written in the loop ---") + ($optInl | ForEach-Object { ($_ -replace '^.*exp8_negative_novec.cpp', 'exp8_negative_novec.cpp') }) |
    Out-File -Encoding utf8 (Join-Path $out "exp8_evidence.txt")

Remove-Item "$out\*.exe" -ErrorAction SilentlyContinue
Write-Output "vectorization evidence written to $out"
Get-Content (Join-Path $out "summary_table.md")
