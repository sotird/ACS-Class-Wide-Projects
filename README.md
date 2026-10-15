# GRADER, READ THIS.

**The appendices at the end of each report hold a tablulated summary of results, a clickable rubric navigates to pertinent sections, and a clickable checklist that shows where to find the project features in terms of the handout. Hopefully this makes your life easier.**

See AI use notes for why.

# ACS Class-wide Projects

Coursework for RPI ECSE 4320/6320 Advanced Computer Systems: three measurement projects. Each project consists of C++ benchmarks, build/run/plotting scripts and a report.


## Layout

| path | content |
|---|---|
| [`project1-simd-advantage-profiling/`](project1-simd-advantage-profiling/README.md) | scalar vs. SIMD speedup of simple numeric kernels |
| [`project2-cache-memory-profiling/`](project2-cache-memory-profiling/README.md) | cache and memory latency, bandwidth, cache/TLB behaviour |
| [`project3-ssd-performance-profiling/`](project3-ssd-performance-profiling/README.md) | SSD performance with FIO and an independent C++ tool |
| [`tools/`](tools/README.md) | shared tooling: hardware-counter recording on Windows (projects 1 and 2) |

## Structure of a project directory

All three projects follow the same layout:

```
project*/
  README.md        layout of that project
  src/             C++ sources, one program per experiment with shared headers
  scripts/         build, run and processing scripts
  data/raw/        measurements exactly as the experiments wrote them
  data/processed/  tables derived from the raw data
  results/         figures used by the report, recorded environment and build information
  report/          report.tex, report.pdf
  bin/             build output
```

The scripts build the experiments, run them, and process and plot the data; the report is compiled from the generated tables and figures. The commands are in Section 1 of each report.
