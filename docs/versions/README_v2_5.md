# SAGE v2.5 - Memory Consolidation Organ

## Summary

v2.5 adds bounded automatic memory consolidation.

Files:

```text
sage_core/memory_consolidation.py
sage_runtime/memory_consolidation_runtime.py
tools/consolidate_memory.py
benchmarks/benchmark_v2_5_memory_consolidation.py
configs/memory_consolidation.json
configs/memory_consolidation_runtime.json
docs/SAGE_v2_5_MEMORY_CONSOLIDATION_ORGAN.md
docs/versions/README_v2_5.md
```

## Run

```powershell
python tools/consolidate_memory.py
python tools/consolidate_memory.py --json
python -m benchmarks.benchmark_v2_5_memory_consolidation
```

## Commit

```powershell
git add sage_core/memory_consolidation.py
git add sage_runtime/memory_consolidation_runtime.py
git add tools/consolidate_memory.py
git add benchmarks/benchmark_v2_5_memory_consolidation.py
git add configs/memory_consolidation.json
git add configs/memory_consolidation_runtime.json
git add docs/SAGE_v2_5_MEMORY_CONSOLIDATION_ORGAN.md
git add docs/versions/README_v2_5.md
git add results/v2_5_memory_consolidation_report.json
git add results/v2_5_memory_consolidation_smoke_result.json

git commit -m "Add SAGE v2.5 memory consolidation organ"
git tag v2.5
```
