# Builds the C++ reference/pitfall tool with MinGW-w64 GCC (the FIO experiments need no build).
$ErrorActionPreference = "Stop"
$env:Path = "C:\msys64\mingw64\bin;" + $env:Path
$root = Split-Path -Parent $PSScriptRoot
New-Item -ItemType Directory -Force -Path (Join-Path $root "bin") | Out-Null
& "C:\msys64\mingw64\bin\g++.exe" -O2 -std=c++17 -static -o (Join-Path $root "bin\ref_qd1.exe") (Join-Path $root "src\ref_qd1.cpp")
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Output "built bin\ref_qd1.exe"
