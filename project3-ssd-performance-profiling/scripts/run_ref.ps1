# Runs the C++ reference tool (independent Win32 synchronous QD=1 I/O) against the
# preconditioned scratch file on T:. 3 trials per mode, 10 s each, interleaved.
# Only touches T:\ssdbench (the tool itself refuses any other drive).
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$exe = Join-Path $root "bin\ref_qd1.exe"
$refdir = Join-Path $root "data\raw\ref"
New-Item -ItemType Directory -Force -Path $refdir | Out-Null
$file = "T:\ssdbench\fio.dat"
$vdl = "T:\ssdbench\vdl.dat"
$modes = @("randread4k_direct", "randwrite4k_direct", "randwrite4k_wt", "seqread128k_direct", "seqwrite128k_direct", "randread4k_buffered", "randwrite4k_buffered")
for ($t = 0; $t -lt 3; $t++) {
    foreach ($m in $modes) {
        & $exe $m $file 10 1024 (Join-Path $refdir "ref_trial$t.csv")
        if ($LASTEXITCODE -ne 0) { throw "ref_qd1 failed: $m" }
        Start-Sleep -Seconds 2
    }
    # unwritten-file pitfall: 64 MiB scratch file (fio.dat occupies 1500 MiB; ~144 MiB remain free on T:)
    & $exe vdl $vdl 8 64 (Join-Path $refdir "ref_trial$t.csv")
    if ($LASTEXITCODE -ne 0) { throw "ref_qd1 vdl failed" }
    Remove-Item $vdl -ErrorAction SilentlyContinue
}
Write-Output "reference runs complete -> $refdir"
