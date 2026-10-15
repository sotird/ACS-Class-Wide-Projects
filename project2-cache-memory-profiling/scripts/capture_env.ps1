# Records the measurement environment to results\environment.txt:
# hardware, OS, power plan / boost policy, measured clock, tool versions, and
# a short idle-load snapshot (isolation check). READ-ONLY: changes nothing.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
$bin = Join-Path $root "bin"
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null
$out = Join-Path $root "results\environment.txt"
$env:Path = "C:\msys64\mingw64\bin;" + $env:Path

$lines = New-Object System.Collections.Generic.List[string]
function Add($s) { $lines.Add($s) | Out-Null }

Add "captured: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$cpu = Get-CimInstance Win32_Processor
Add "cpu: $($cpu.Name.Trim())"
Add "cores/threads: $($cpu.NumberOfCores)/$($cpu.NumberOfLogicalProcessors)  (SMT $(if ($cpu.NumberOfLogicalProcessors -gt $cpu.NumberOfCores) {'ENABLED'} else {'disabled'}))"
Add "windows-reported base clock (MHz): $($cpu.MaxClockSpeed)   [NOT the sustained boost clock -- see measured clock]"
Add "L2 total (KB): $($cpu.L2CacheSize)   L3 total (KB): $($cpu.L3CacheSize)"
Get-CimInstance Win32_PhysicalMemory | ForEach-Object { Add "dimm: $($_.PartNumber.Trim()) $([math]::Round($_.Capacity/1GB))GB configured $($_.ConfiguredClockSpeed) MT/s" }
$os = Get-CimInstance Win32_OperatingSystem
Add "os: $($os.Caption) $($os.Version) build $($os.BuildNumber)"
Add "power plan: $((powercfg /getactivescheme) -replace '^Power Scheme GUID: ','')"
foreach ($setting in @("PERFBOOSTMODE", "PROCTHROTTLEMIN", "PROCTHROTTLEMAX")) {
    $q = powercfg /query SCHEME_CURRENT SUB_PROCESSOR $setting 2>$null
    $m = $q | Select-String "Current AC Power Setting Index"
    Add "  ${setting}: $(if ($m) { $m.ToString().Trim() } else { 'not exposed by this power scheme' })"
}
Add "admin: $(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"
Add "g++: $((& 'C:\msys64\mingw64\bin\g++.exe' --version | Select-Object -First 1))"
Add "python: $((& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version) 2>&1)"
$fc = Join-Path $bin "freq_calibrate.exe"
if (Test-Path $fc) {
    $ghz = & $fc 5
    Add "measured sustained clock on pinned CPU (GHz, dependent-add chain, 5 samples): $($ghz -join ', ')"
}
# idle-load snapshot: if the machine is not quiet, results are suspect
$cpuLoad = (Get-Counter '\Processor(_Total)\% Processor Time' -SampleInterval 1 -MaxSamples 5 | ForEach-Object { $_.CounterSamples.CookedValue } | Measure-Object -Average).Average
Add "background CPU load before run (avg of 5 x 1s, %): $([math]::Round($cpuLoad,2))"
Add "top background processes by CPU time:"
Get-Process | Sort-Object CPU -Descending | Select-Object -First 5 | ForEach-Object { Add "  $($_.ProcessName)  cpu=$([math]::Round($_.CPU,1))s" }

$lines | Out-File -Encoding utf8 $out
$lines | ForEach-Object { Write-Output $_ }
