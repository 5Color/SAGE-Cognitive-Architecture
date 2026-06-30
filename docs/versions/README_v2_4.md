# SAGE v2.4 - CPU Language Core Architecture Probe

## Summary

v2.4 adds a CPU-only language core architecture probe.

Files:

```text
sage_core/cpu_language_core.py
sage_runtime/cpu_language_core_runtime.py
tools/run_cpu_language_core.py
benchmarks/benchmark_v2_4_cpu_language_core.py
configs/cpu_language_core.json
docs/SAGE_v2_4_CPU_LANGUAGE_CORE_ARCHITECTURE_PROBE.md
docs/versions/README_v2_4.md
```

## Run

```powershell
python tools/run_cpu_language_core.py --demo
python tools/run_cpu_language_core.py --text "SAGE 현재 진행상황 요약해줘"
python -m benchmarks.benchmark_v2_4_cpu_language_core
```

## Commit

```powershell
git add sage_core/cpu_language_core.py
git add sage_runtime/cpu_language_core_runtime.py
git add tools/run_cpu_language_core.py
git add benchmarks/benchmark_v2_4_cpu_language_core.py
git add configs/cpu_language_core.json
git add docs/SAGE_v2_4_CPU_LANGUAGE_CORE_ARCHITECTURE_PROBE.md
git add docs/versions/README_v2_4.md
git add results/v2_4_cpu_language_core_result.json
git add results/v2_4_cpu_language_core_benchmark.json

git commit -m "Add SAGE v2.4 CPU language core probe"
git tag v2.4
```
