# SAGE v2.8 - Memory Context Manager

## Summary

v2.8 adds query-specific memory context selection.

Files:

```text
sage_core/memory_context_manager.py
sage_runtime/memory_context_runtime.py
tools/build_memory_context.py
benchmarks/benchmark_v2_8_memory_context_manager.py
configs/memory_context_manager.json
configs/memory_context_runtime.json
docs/SAGE_v2_8_MEMORY_CONTEXT_MANAGER.md
docs/versions/README_v2_8.md
```

## Run

```powershell
python tools/build_memory_context.py --query "SAGE 현재 진행상황과 다음 단계 알려줘"
python -m benchmarks.benchmark_v2_8_memory_context_manager
```

## Commit

```powershell
git add sage_core/memory_context_manager.py
git add sage_runtime/memory_context_runtime.py
git add tools/build_memory_context.py
git add benchmarks/benchmark_v2_8_memory_context_manager.py
git add configs/memory_context_manager.json
git add configs/memory_context_runtime.json
git add docs/SAGE_v2_8_MEMORY_CONTEXT_MANAGER.md
git add docs/versions/README_v2_8.md
git add results/v2_8_memory_context_bundle.json
git add results/v2_8_memory_context_manager_smoke_result.json

git commit -m "Add SAGE v2.8 memory context manager"
git tag v2.8
```
