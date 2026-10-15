# Records 1-second OS-level disk and CPU counters (and the busiest I/O processes) while a benchmark runs, so that
# throughput changes can be checked against other activity on the physical disk and against host-CPU saturation
# (interrupt / DPC time on the logical CPU that services the SATA controller). Read-only; needs no admin rights.
param([int]$Seconds = 300, [string]$Out = "os_monitor.csv")
$sets = @(
  '\PhysicalDisk(0 *)\Disk Reads/sec', '\PhysicalDisk(0 *)\Disk Writes/sec',
  '\PhysicalDisk(0 *)\Avg. Disk sec/Read', '\PhysicalDisk(0 *)\Current Disk Queue Length',
  '\Processor(_Total)\% Processor Time',
  '\Processor Information(*)\% DPC Time', '\Processor Information(*)\% Interrupt Time', '\Processor Information(*)\% Processor Time'
)
$rows = New-Object System.Collections.Generic.List[object]
$t0 = Get-Date
for ($i = 0; $i -lt $Seconds; $i++) {
  $c = Get-Counter -Counter $sets -SampleInterval 1 -MaxSamples 1 -ErrorAction SilentlyContinue
  $o = [ordered]@{ t = [math]::Round(((Get-Date) - $t0).TotalSeconds, 1) }
  $pi = @{}
  foreach ($s in $c.CounterSamples) {
    if ($s.Path -like '*processor information*') {
      if ($s.InstanceName -match '_total') { continue }
      $k = $s.Path -replace '^.*\\', ''; if (-not $pi.ContainsKey($k)) { $pi[$k] = @{} }; $pi[$k][$s.InstanceName] = $s.CookedValue
    } else { $o[($s.Path -replace '^.*\\', '')] = [math]::Round($s.CookedValue, 4) }
  }
  foreach ($k in '% dpc time', '% interrupt time', '% processor time') {
    if ($pi.ContainsKey($k)) {
      $m = $pi[$k].GetEnumerator() | Sort-Object Value -Descending | Select-Object -First 1
      $o["max cpu $k"] = [math]::Round($m.Value, 1); $o["cpu of max $k"] = $m.Name
    }
  }
  $rows.Add([pscustomobject]$o)
  if ($i % 10 -eq 0) {   # who else touches the disk? (every 10 s, top 3 by I/O operations/sec, excluding fio)
    $p = Get-Counter '\Process(*)\IO Data Operations/sec' -ErrorAction SilentlyContinue
    $top = $p.CounterSamples | Where-Object { $_.InstanceName -notin '_total', 'idle', 'fio' -and $_.CookedValue -gt 20 } |
           Sort-Object CookedValue -Descending | Select-Object -First 3 | ForEach-Object { "{0}={1:N0}" -f $_.InstanceName, $_.CookedValue }
    Add-Content -Path ($Out + ".procs.txt") -Value ("{0}s: {1}" -f $o.t, ($top -join ", "))
  }
}
$rows | Export-Csv -Path $Out -NoTypeInformation
