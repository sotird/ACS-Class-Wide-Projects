# Records hardware performance counters (PMCs) while ONE program runs and prints the totals attributed to that
# program's process.  Needs an ELEVATED PowerShell (ETW kernel trace).  Uses WPR (built into Windows) to record the
# counters on every context switch and tools\etl_pmc.exe to read them back (see etl_pmc.cpp).
#
#   .\pmc_record.ps1 -Exe path\to\prog.exe -ExeArgs "a b c" -Counters TotalIssues,DcacheMisses
#       -Custom "DtlbMissAll=0x45:0xFF","DramFills=0x44:0x48"     # extra AMD events as Name=Event:UnitMask (hex)
#       -OutCsv result.csv                                         # one row of counter totals for the program
#
# Notes: at most 6 counters can be recorded at once (Zen 5 has 6 programmable PMCs); the counters of one trace record come back
# in alphabetical order of their NAMES, not the order requested (this script sorts and maps accordingly); custom events are
# registered through a MicroArchitecturalConfig with Architecture="AMD" (event code and unit mask as in AMD's event tables).
# Counter names come from `wpr -pmcsources`.  Only the counters of the logical CPU the program ran on are read, so
# pin the program (all benchmarks here pin themselves).  No system setting is changed: the trace session ends with the run.
param([Parameter(Mandatory = $true)][string]$Exe, [string]$ExeArgs = "", [string[]]$Counters = @("DcacheMisses", "CacheMisses"),
      [string[]]$Custom = @(), [string]$OutCsv = "", [string]$Tag = "run", [string]$WorkDir = $env:TEMP, [string]$ProfileInterval = "0x10000")
$ErrorActionPreference = "Continue"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$parser = Join-Path $here "etl_pmc.exe"
$wprp = Join-Path $WorkDir "pmc_$Tag.wprp"; $etl = Join-Path $WorkDir "pmc_$Tag.etl"
Remove-Item $etl -ErrorAction SilentlyContinue
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1
if ($cpu.Description -match "Family (\d+) Model (\d+)") { $fam = $Matches[1]; $mod = $Matches[2] } else { $fam = ""; $mod = "" }
$cfg = ""; $names = @($Counters)
if ($Custom.Count -gt 0) {
    $ps = ""
    foreach ($c in $Custom) {
        $n, $eu = $c -split "=", 2; $ev, $un = $eu -split ":", 2
        $ps += "        <ProfileSource Name=`"$n`" Event=`"$ev`" Unit=`"$un`" Interval=`"$ProfileInterval`" AllowsHalt=`"false`" Persist=`"false`"/>`n"
        $names += $n
    }
    $cfg = "    <MicroArchitecturalConfig Id=`"CounterConfig_Custom`" Base=`"`">`n      <ProfileSources Architecture=`"AMD`" Family=`"$fam`" Model=`"$mod`" Description=`"custom AMD events`">`n$ps      </ProfileSources>`n    </MicroArchitecturalConfig>`n"
}
# Windows returns the counters of one context-switch record in ORDINAL ALPHABETICAL order of their names (verified
# empirically with single-counter runs), so list them sorted and map the results by that order.
$names = [string[]]$names; [Array]::Sort($names, [StringComparer]::Ordinal)
$ctr = ($names | ForEach-Object { "        <Counter Value=`"$_`"/>" }) -join "`n"
$cfgRef = if ($Custom.Count -gt 0) { "      <MicroArchitecturalConfigId Value=`"CounterConfig_Custom`"/>`n" } else { "" }
@"
<?xml version="1.0" encoding="utf-8"?>
<WindowsPerformanceRecorder Version="1.0">
  <Profiles>
    <SystemCollector Id="SystemCollector" Name="NT Kernel Logger"><BufferSize Value="1024"/><Buffers Value="64"/></SystemCollector>
    <SystemProvider Id="SystemProvider"><Keywords><Keyword Value="ProcessThread"/><Keyword Value="Loader"/><Keyword Value="CSwitch"/></Keywords></SystemProvider>
$cfg    <HardwareCounter Id="HC_PMC" Base="" Strict="true">
$cfgRef      <Counters>
$ctr
      </Counters>
      <Events><Event Value="CSwitch"/></Events>
    </HardwareCounter>
    <Profile Id="PMC.Verbose.File" Name="PMC" Description="PMC on CSwitch" LoggingMode="File" DetailLevel="Verbose">
      <Collectors><SystemCollectorId Value="SystemCollector"><SystemProviderId Value="SystemProvider"/><HardwareCounterId Value="HC_PMC"/></SystemCollectorId></Collectors>
    </Profile>
  </Profiles>
</WindowsPerformanceRecorder>
"@ | Out-File -Encoding utf8 $wprp
wpr -cancel 2>&1 | Out-Null
$o = wpr -start "$wprp!PMC" -filemode 2>&1; if ($LASTEXITCODE -ne 0) { throw "wpr -start failed: $o" }
try {
    $pi = New-Object System.Diagnostics.ProcessStartInfo; $pi.FileName = $Exe; $pi.Arguments = $ExeArgs; $pi.UseShellExecute = $false
    $pi.RedirectStandardOutput = $true; $pi.RedirectStandardError = $true
    $p = [System.Diagnostics.Process]::Start($pi); $outTask = $p.StandardOutput.ReadToEndAsync(); $errTask = $p.StandardError.ReadToEndAsync()
    $p.WaitForExit(); $stdout = $outTask.Result; $pid_ = $p.Id
} finally { $o = wpr -stop $etl "pmc" 2>&1 }
if (-not (Test-Path $etl)) { throw "no ETL produced: $o" }
$tbl = & $parser $etl | ConvertFrom-Csv
$row = $tbl | Where-Object { [int64]$_.pid -eq $pid_ } | Select-Object -First 1
if (-not $row) { throw "process $pid_ not found in trace" }
$res = [ordered]@{ tag = $Tag; pid = $pid_; switches = $row.switches }
for ($i = 0; $i -lt $names.Count; $i++) { $res[$names[$i]] = [int64]$row."counter$i" }
$res.Add("program_stdout", ($stdout -replace "[\r\n]+", " | ").Trim())
$obj = [pscustomobject]$res
if ($OutCsv) { $obj | Export-Csv -Path $OutCsv -NoTypeInformation -Append }
Remove-Item $etl, $wprp -ErrorAction SilentlyContinue
$obj
