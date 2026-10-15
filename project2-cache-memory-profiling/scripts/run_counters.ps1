# Hardware-counter experiments for Project 2 (exp9). Needs an ELEVATED PowerShell (ETW kernel trace) and
# tools\etl_pmc.exe + tools\pmc_record.ps1 (see tools\README.md). Records, for each configuration, the counters of
# the pinned target process (cycles, retired instructions, L1D fills from L2 / L3 / DRAM, DTLB misses) and
# DIFFERENCES two runs with different iteration counts, so allocation/initialisation cancels out:
#     counts per unit = (C(run2) - C(run1)) / (units2 - units1)
# Writes data\raw\counters\exp9_counters.csv.  Nothing system-wide is changed (the trace session ends with each run).
param([string[]]$Only = @())     # e.g. -Only E2_discipline re-measures just that experiment group and replaces its rows
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$tools = Join-Path (Split-Path -Parent $root) "tools"
$rec = Join-Path $tools "pmc_record.ps1"
$exe = Join-Path $root "bin\exp9_counter_target.exe"
$outdir = Join-Path $root "data\raw\counters"; New-Item -ItemType Directory -Force -Path $outdir | Out-Null
$csv = Join-Path $outdir "exp9_counters.csv"
if ($Only.Count -gt 0 -and (Test-Path $csv)) { $keep = Import-Csv $csv | Where-Object { $_.exp -notin $Only }; Remove-Item $csv; $keep | Export-Csv -Path $csv -NoTypeInformation } elseif ($Only.Count -eq 0) { Remove-Item $csv -ErrorAction SilentlyContinue }
$custom = @("FillL2=0x44:0x01", "FillL3=0x44:0x02", "FillDram=0x44:0x48", "DtlbMiss=0x45:0xFF")   # AMD LsAnyFillsFromSys (source masks), LsL1DTlbMiss (all)
$counters = @("TotalCycles", "InstructionRetired")
$work = Join-Path $env:TEMP "exp9_counters"; New-Item -ItemType Directory -Force -Path $work | Out-Null

function Measure-Config([string]$exp, [string]$label, [string[]]$args1, [string[]]$args2, [int]$rep = 1) {
  if ($Only.Count -gt 0 -and $exp -notin $Only) { return }
  for ($k = 1; $k -le $rep; $k++) {
    $r1 = & $rec -Exe $exe -ExeArgs ($args1 -join " ") -Counters $counters -Custom $custom -Tag "${exp}_a" -WorkDir $work 2>$null
    $r2 = & $rec -Exe $exe -ExeArgs ($args2 -join " ") -Counters $counters -Custom $custom -Tag "${exp}_b" -WorkDir $work 2>$null
    $u1 = [double](($r1.program_stdout -split "units=")[1] -split " ")[0]; $u2 = [double](($r2.program_stdout -split "units=")[1] -split " ")[0]
    $ns = [double](($r2.program_stdout -split "ns_per_unit=")[1] -split " ")[0]
    $du = $u2 - $u1
    $row = [ordered]@{ exp = $exp; label = $label; rep = $k; args = ($args2 -join " "); units_run1 = $u1; units_run2 = $u2; ns_per_unit = $ns }
    foreach ($c in ($counters + ($custom | ForEach-Object { ($_ -split "=")[0] }))) { $row["${c}_per_unit"] = [math]::Round(([double]$r2.$c - [double]$r1.$c) / $du, 5) }
    $obj = [pscustomobject]$row; $obj | Export-Csv -Path $csv -NoTypeInformation -Append
    "{0,-12} {1,-26} ns/unit={2,8:N3}  fillL2={3,6:N3} fillL3={4,6:N3} fillDRAM={5,6:N3} dtlb={6,6:N3} cyc={7,7:N2} instr={8,6:N2}" -f $exp, $label, $ns, $obj.FillL2_per_unit, $obj.FillL3_per_unit, $obj.FillDram_per_unit, $obj.DtlbMiss_per_unit, $obj.TotalCycles_per_unit, $obj.InstructionRetired_per_unit
  }
}

# E1: working-set sweep of the dependent chase (compact pages): footprint -> fills by level, TLB, cycles.
# Large footprints need many more hops than the setup (allocation, shuffling, page faults) costs, and 4-64 MiB footprints depend on which physical
# pages the OS happens to hand out (L3 set conflicts), so those are repeated 3x (median taken in process_counters.py).
foreach ($kib in 16, 32, 64, 256, 512, 1024, 4096, 8192, 16384, 32768, 65536, 262144, 1048576) {
    $lines = $kib * 1024 / 64; $pages = [math]::Max(1, [math]::Ceiling($lines / 64))
    if ($kib -ge 262144) { $h = @(20000000, 220000000); $rep = 1 } elseif ($kib -ge 4096) { $h = @(30000000, 150000000); $rep = 3 } else { $h = @(20000000, 60000000); $rep = 1 }
    Measure-Config "E1_chase" ("{0} KiB" -f $kib) @("chase", $lines, $pages, $h[0]) @("chase", $lines, $pages, $h[1]) $rep
}
# E2: the same 1 GiB of lines touched three ways: dependent, independent random, sequential
$L = 16777216
Measure-Config "E2_discipline" "dependent chase" @("chase", $L, 262144, 20000000) @("chase", $L, 262144, 220000000)
Measure-Config "E2_discipline" "independent random" @("rand", $L, 6) @("rand", $L, 36)
Measure-Config "E2_discipline" "sequential" @("seq", $L, 18) @("seq", $L, 90)
# E3: TLB reach at a fixed cache footprint (512 KiB = L2-resident; 32 KiB = L1; 4 MiB = L3): same lines over more pages
foreach ($kib in 32, 512, 4096) {
    $lines = $kib * 1024 / 64
    foreach ($pg in @(($lines / 64), ($lines / 16), ($lines / 4), $lines)) {
        if ($pg -lt 1) { continue }
        if ($kib -eq 4096 -and [int]$pg -eq $lines) { $h = @(20000000, 220000000); $rep = 1 } elseif ($kib -eq 4096) { $h = @(30000000, 150000000); $rep = 3 } else { $h = @(20000000, 60000000); $rep = 1 }
        Measure-Config "E3_tlb" ("{0} KiB over {1} pages" -f $kib, [int]$pg) @("chase", $lines, [int]$pg, $h[0]) @("chase", $lines, [int]$pg, $h[1]) $rep
    }
}
# E5: 2 MiB huge pages vs 4 KiB pages, same lines and the same randomly permuted page slots (pmode 1 = huge pages, 2 = 4 KiB control).
# Needs SeLockMemoryPrivilege ("Lock pages in memory", granted by the machine's owner); the target process only switches it on in its own token.
foreach ($kib in 512, 4096) {
    $lines = $kib * 1024 / 64
    foreach ($pg in @(($lines / 64), ($lines / 4), $lines)) {
        foreach ($pm in 2, 1) {
            if ($kib -eq 4096) { $h = @(30000000, 150000000); $rep = 3 } else { $h = @(20000000, 60000000); $rep = 1 }
            Measure-Config "E5_hugepages" ("{0} KiB over {1} pages, {2}" -f $kib, [int]$pg, $(if ($pm -eq 1) { "2 MiB pages" } else { "4 KiB pages" })) @("chase", $lines, [int]$pg, $h[0], $pm) @("chase", $lines, [int]$pg, $h[1], $pm) $rep
        }
    }
}
# E4: simple loop kernels (the course document's z=x*y and y=a*x+y), float32, 3 arrays of n elements
foreach ($k in "saxpy", "mul") {
    foreach ($n in 4096, 65536, 1048576, 33554432) {
        $r = if ($n -le 65536) { @(200000, 600000) } elseif ($n -le 1048576) { @(4000, 12000) } else { @(30, 90) }
        Measure-Config "E4_$k" ("n=$n") @($k, $n, $r[0]) @($k, $n, $r[1])
    }
}
Write-Output "wrote $csv"
