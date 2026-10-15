# Shared tools

Tools shared by the projects. Usage notes are in the header comment of each file; the counter pipeline's method
and findings are described in the reports of project 1 (Sec. 2.3 and 4.6) and project 2 (Sec. 2.3, 4.5 and 4.6).

## Hardware performance counters on Windows

Windows has no `perf`; these tools read the CPU's performance counters through ETW using only components that ship with Windows.

| file | role |
|---|---|
| `pmc_record.ps1` | records the counters of one program: starts a WPR session, runs the program, stops the session, and prints the totals attributed to that program (elevated PowerShell) |
| `etl_pmc.cpp` | trace consumer: reads the counter values from the recorded `.etl` file (`etl_pmc.exe` is its build) |
| `pmc_validate.cpp` | dependent pointer chase with a known miss behaviour, used to validate the pipeline (`pmc_validate.exe` is its build) |
