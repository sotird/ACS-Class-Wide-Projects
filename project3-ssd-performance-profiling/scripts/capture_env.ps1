# Records the SSD measurement environment to results\environment.txt:
# drive model/interface/firmware, partition + alignment, filesystem, volume
# free space, write-cache and power state where visible without admin, host
# CPU/OS, FIO version, background activity. READ-ONLY: changes nothing.
$ErrorActionPreference = "Continue"
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $root "results") | Out-Null
$out = Join-Path $root "results\environment.txt"
$L = New-Object System.Collections.Generic.List[string]
function Add($s) { $L.Add($s) | Out-Null }
Add "captured: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
$cpu = Get-CimInstance Win32_Processor
Add "host cpu: $($cpu.Name.Trim()) ($($cpu.NumberOfCores)C/$($cpu.NumberOfLogicalProcessors)T)"
$os = Get-CimInstance Win32_OperatingSystem
Add "os: $($os.Caption) $($os.Version) build $($os.BuildNumber)   RAM: $([math]::Round($os.TotalVisibleMemorySize/1MB,1)) GiB"
Add "power plan: $((powercfg /getactivescheme) -replace '^Power Scheme GUID: ','')"
$part = Get-Partition -DriveLetter T
$disk = Get-Disk -Number $part.DiskNumber
$pd = Get-PhysicalDisk -DeviceNumber $part.DiskNumber
Add "ssd: $($disk.FriendlyName)  interface: $($disk.BusType)  media: $($pd.MediaType)  firmware: $($disk.FirmwareVersion)  health: $($disk.HealthStatus)"
Add "ssd capacity: $([math]::Round($disk.Size/1GB,1)) GB; sector sizes logical/physical: $($disk.LogicalSectorSize)/$($disk.PhysicalSectorSize)"
Add "test partition T: size $([math]::Round($part.Size/1MB,1)) MiB, offset $($part.Offset) bytes (offset mod 4096 = $($part.Offset % 4096), mod 1MiB = $($part.Offset % 1048576))"
Add "other partitions on the same disk (NEVER opened by the benchmark): $((Get-Partition -DiskNumber $part.DiskNumber | Where-Object { $_.PartitionNumber -ne $part.PartitionNumber -and $_.DriveLetter } | ForEach-Object { "$($_.DriveLetter): $([math]::Round($_.Size/1GB))GB" }) -join ', ')"
$vol = Get-Volume -DriveLetter T
Add "volume T: label '$($vol.FileSystemLabel)' $($vol.FileSystem), size $([math]::Round($vol.Size/1MB,1)) MiB, free $([math]::Round($vol.SizeRemaining/1MB,1)) MiB"
$wv = Get-CimInstance Win32_Volume -Filter "DriveLetter='T:'"
Add "ntfs cluster size: $($wv.BlockSize) bytes   indexing: $($wv.IndexingEnabled)   compressed: $($wv.Compressed)"
try { $adv = Get-PhysicalDisk -DeviceNumber $part.DiskNumber | Get-StorageAdvancedProperty -ErrorAction Stop; Add "device cache: IsDeviceCacheEnabled=$($adv.IsDeviceCacheEnabled) IsPowerProtected=$($adv.IsPowerProtected)" } catch { Add "device write-cache state: not readable without administrator rights (assumed Windows default: enabled)" }
try { $rc = Get-PhysicalDisk -DeviceNumber $part.DiskNumber | Get-StorageReliabilityCounter -ErrorAction Stop; Add "temperature: $($rc.Temperature) C  wear: $($rc.Wear)  power-on hours: $($rc.PowerOnHours)" } catch { Add "temperature / SMART wear: NOT readable (needs administrator rights) -- thermal throttling cannot be directly observed in this setup" }
try { $mp = Get-MpComputerStatus -ErrorAction Stop; Add "windows defender real-time protection: $($mp.RealTimeProtectionEnabled)" } catch { Add "windows defender status: unavailable" }
$av = Get-Process | Where-Object { $_.ProcessName -match 'Malwarebytes|MsMpEng|NordVPN' } | Select-Object -ExpandProperty ProcessName -Unique
Add "security/other background software running: $($av -join ', ')"
$fio = @("$env:LOCALAPPDATA\Programs\fio-extracted\fio\fio.exe", "C:\Program Files\fio\fio.exe") | Where-Object { Test-Path $_ } | Select-Object -First 1
if ($fio) { Add "fio: $(& $fio --version)  ($fio)" } else { Add "fio: not installed" }
Add "python: $((& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" --version) 2>&1)"
Add "g++: $((& 'C:\msys64\mingw64\bin\g++.exe' --version | Select-Object -First 1))"
Add "admin rights: $(([Security.Principal.WindowsPrincipal][Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator))"
$c = Get-Counter '\PhysicalDisk(0 *)\Disk Transfers/sec','\PhysicalDisk(0 *)\Disk Bytes/sec' -SampleInterval 1 -MaxSamples 10
$tr = ($c | ForEach-Object { $_.CounterSamples[0].CookedValue } | Measure-Object -Average).Average
$by = ($c | ForEach-Object { $_.CounterSamples[1].CookedValue } | Measure-Object -Average).Average
Add "background activity on the SSD before the run (10 x 1s): $([math]::Round($tr,1)) transfers/s, $([math]::Round($by/1024,1)) KiB/s"
$L | Out-File -Encoding utf8 $out
$L | ForEach-Object { Write-Output $_ }
