# Hardware-counter evidence for Project 1 (exp9): instructions retired, cycles, loads and cache-line fills per element
# for each kernel x ISA variant x working-set size.  Needs an ELEVATED PowerShell (ETW kernel trace) and tools\etl_pmc.exe
# + tools\pmc_record.ps1.  Two runs per configuration with different call counts are DIFFERENCED so that allocation and
# initialisation cancel:  counts per element = (C(run2) - C(run1)) / (elements2 - elements1).
# Writes data\raw\counters\exp9_counters.csv.  Nothing system-wide is changed (the trace session ends with each run).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path (Split-Path -Parent $root) "tools"
$rec = Join-Path $tools "pmc_record.ps1"
$outdir = Join-Path $root "data\raw\counters"; New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$csv = Join-Path $outdir "exp9_counters.csv"; Remove-Item $csv -ErrorAction SilentlyContinue
$custom = @("FillL2=0x44:0x01", "FillL3=0x44:0x02", "FillDram=0x44:0x48")   # AMD LsAnyFillsFromSys (L1D fills by source: L2, L3, DRAM)
$counters = @("TotalCycles", "InstructionRetired", "DcacheAccesses")
$work = Join-Path $env:TEMP "p1_counters"; New-Item -ItemType Directory -Force -Path $work | Out-Null

function Measure-Config([string]$variant, [string]$kernel, [int]$n, [int]$c1, [int]$c2) {
    $exe = Join-Path $root "bin\exp9_counters_$variant.exe"
    $r1 = & $rec -Exe $exe -ExeArgs "$kernel $n $c1" -Counters $counters -Custom $custom -Tag "p1_a" -WorkDir $work 2>$null
    $r2 = & $rec -Exe $exe -ExeArgs "$kernel $n $c2" -Counters $counters -Custom $custom -Tag "p1_b" -WorkDir $work 2>$null
    $u1 = [double](($r1.program_stdout -split "units=")[1] -split " ")[0]; $u2 = [double](($r2.program_stdout -split "units=")[1] -split " ")[0]
    $ns = [double](($r2.program_stdout -split "ns_per_unit=")[1] -split " ")[0]
    $du = $u2 - $u1
    $row = [ordered]@{ variant = $variant; kernel = $kernel; n = $n; calls_run2 = $c2; ns_per_elem = $ns }
    foreach ($c in ($counters + ($custom | ForEach-Object { ($_ -split "=")[0] }))) { $row["${c}_per_elem"] = [math]::Round(([double]$r2.$c - [double]$r1.$c) / $du, 6) }
    $obj = [pscustomobject]$row; $obj | Export-Csv -Path $csv -NoTypeInformation -Append
    "{0,-10} {1,-8} n={2,9}  ns/elem={3,8:N4} cyc={4,6:N3} instr={5,6:N3} loads={6,6:N3} fillL2={7,7:N4} fillL3={8,7:N4} fillDRAM={9,7:N4}" -f $variant, $kernel, $n, $ns, $obj.TotalCycles_per_elem, $obj.InstructionRetired_per_elem, $obj.DcacheAccesses_per_elem, $obj.FillL2_per_elem, $obj.FillL3_per_elem, $obj.FillDram_per_elem
}

# footprints: n = 2048 (3 arrays x 8 KiB = 24 KiB: L1), 65536 (768 KiB: L2), 1048576 (12 MiB: L3), 33554432 (384 MiB: DRAM)
$sizes = @(@(2048, 400000, 1200000), @(65536, 12000, 36000), @(1048576, 800, 2400), @(33554432, 40, 160))
foreach ($variant in "scalar", "sse2", "avx2", "avx512") {
    foreach ($k in "saxpy", "mul", "stencil", "dot") {
        foreach ($s in $sizes) { Measure-Config $variant $k $s[0] $s[1] $s[2] }
    }
}
foreach ($s in $sizes) { Measure-Config "avx512_fm" "dot" $s[0] $s[1] $s[2] }
Write-Output "wrote $csv"
